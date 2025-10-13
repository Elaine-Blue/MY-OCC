import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops import knn, Voxelization
from mmdet.core import multi_apply
from mmdet.models import HEADS
from mmdet.models import builder
from mmdet.models.builder import build_loss
from .bbox.utils import decode_points
import pickle
import copy

@HEADS.register_module()
class SSDOCCHead(BaseModule):
    def __init__(self,
                 num_classes,
                 pc_range=[],
                 empty_label=17,
                 occ_names=[],
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 branch_loss_weights=[1.0, 1.0],
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_pts=dict(type='L1Loss'),
                 region_aware_branch=None,
                 object_aware_branch=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.occ_names = occ_names
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.empty_label = empty_label
        self.branch_loss_weights = branch_loss_weights
        self.loss_cls = build_loss(loss_cls)
        self.loss_pts = build_loss(loss_pts)
        if region_aware_branch is not None:
            self.region_aware_branch = builder.build_head(region_aware_branch)
        if object_aware_branch is not None:
            self.object_aware_branch = builder.build_head(object_aware_branch)
        
        self.num_query = self.region_aware_branch.num_query + self.object_aware_branch.num_query
        self.last_layer_num_refine = self.region_aware_branch.num_refines[-1]
        self.scene_size = self.region_aware_branch.scene_size
        self.voxel_size = self.region_aware_branch.voxel_size
        self.pc_range = self.region_aware_branch.pc_range
        self.voxel_num = self.region_aware_branch.voxel_num
        self.matched_results = {}
    
        self.voxel_generator = Voxelization(
            self.voxel_size,
            self.pc_range,
            max_num_points=10, 
            max_voxels=self.num_query * self.last_layer_num_refine,
            deterministic=False
        )
    
    def init_weights(self):
        self.region_aware_branch.init_weights()
        self.object_aware_branch.init_weights()

    def forward(self, mlvl_feats, img_metas):
        mlvl_feats_r = [mlvl_feats[0], mlvl_feats[2]]
        mlvl_feats_o = [mlvl_feats[1], mlvl_feats[3]]
        result_dict = {}
        
        init_points_r, cls_scores_r, refine_pts_r = self.region_aware_branch(mlvl_feats_r, img_metas)
        init_points_o, cls_scores_o, refine_pts_o = self.object_aware_branch(mlvl_feats_o, img_metas)
        
        # TODO: Merge two branches output
        result_dict['init_points'] = torch.concat([init_points_r, init_points_o], dim=1)
        result_dict['all_cls_scores_r'] = cls_scores_r
        result_dict['all_refine_pts_r'] = refine_pts_r
        result_dict['all_cls_scores_o'] = cls_scores_o
        result_dict['all_refine_pts_o'] = refine_pts_o
        return result_dict

    def get_dis_weight(self, pts):
        device = pts.device
        max_dist = torch.sqrt(
            self.scene_size[0] ** 2 + self.scene_size[1] ** 2)
        centers = ((self.pc_range[:3] + self.pc_range[3:]) / 2).to(device)
        dist = (pts - centers[None, ...])[..., :2]
        dist = torch.norm(dist, dim=-1)
        return dist / max_dist + 1
    
    def discretize(self, pts, clip=True, decode=False):
        loc = torch.floor((pts - self.pc_range[:3]) / self.voxel_size)
        if clip:
            loc[..., 0] = loc[..., 0].clamp(0, self.voxel_num[0] - 1)
            loc[..., 1] = loc[..., 1].clamp(0, self.voxel_num[1] - 1)
            loc[..., 2] = loc[..., 2].clamp(0, self.voxel_num[2] - 1)

        return loc.long() if not decode else \
            (loc + 0.5) * self.voxel_size + self.pc_range[:3]

    @torch.no_grad()
    def _get_target_single(self, refine_pts, gt_points, gt_masks, gt_labels, cls_weights):
        # For ObjectAware branch, exists no gt_labels case.
        # To avoid compute error, set matched results to empty tensor.
        if len(gt_points) == 0 or len(gt_labels) == 0:
            device = refine_pts.device
            empty_tensor = torch.tensor([], device=device)
            return empty_tensor.to(torch.long), [], [], empty_tensor, empty_tensor
        
        # knn to apply Chamfer distance
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze(0).squeeze(1).long()
        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze(0).squeeze(1).long()
        gt_paired_pts = refine_pts[gt_paired_idx]
        pred_paired_pts = gt_points[pred_paired_idx]
        refine_pts_labels = gt_labels[pred_paired_idx]
        cls_weights = refine_pts.new_tensor(cls_weights)
        label_weights = cls_weights * \
            self.get_dis_weight(pred_paired_pts)[..., None]

        # gt side assignment
        empty_dist_thr = self.train_cfg.get('empty_dist_thr', 0.2)
        empty_weights = self.train_cfg.get('empty_weights', 5)

        gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        dist = torch.norm(gt_points - gt_paired_pts, dim=-1)
        mask = (dist > empty_dist_thr) & gt_masks
        gt_pts_weights[mask] = empty_weights
        return (refine_pts_labels, gt_paired_idx, pred_paired_idx, 
                label_weights, gt_pts_weights)
    
    def get_targets(self):
        # To instantiate the abstract method
        pass

    def loss_single(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_masks_list,
                    gt_labels_list,
                    cls_weights=None):
        
        batch_size, _, refine_num, num_classes = cls_scores.shape # B, Q, N, C
        cls_scores = cls_scores.reshape(batch_size, -1, num_classes)
        refine_pts = refine_pts.reshape(batch_size, -1, 3)
            
        refine_pts = decode_points(refine_pts, self.pc_range)
        cls_scores_list = [cls_scores[i] for i in range(batch_size)]
        refine_pts_list = [refine_pts[i] for i in range(batch_size)]
        cls_weights_list = [cls_weights for _ in range(batch_size)]
        
        # Process Each Sample
        (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights,
         gt_pts_weights) = multi_apply(self._get_target_single, refine_pts_list, gt_points_list, 
                        gt_masks_list, gt_labels_list, cls_weights_list)
        
        gt_paired_pts, pred_paired_pts, valid_cls_scores, valid_refine_pts, valid_labels_list = [], [], [], [], []
        for i in range(batch_size):
            if gt_paired_idx_list[i] == []:
                continue
            
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

            valid_labels_list.append(labels_list[i])
            valid_cls_scores.append(cls_scores_list[i])
            valid_refine_pts.append(refine_pts_list[i])
            
        # Record Matched Results
        if refine_num == self.last_layer_num_refine:
            last_matched_result = dict(
                gt_paired_pts=pred_paired_pts,
                pred_paired_pts=valid_refine_pts,
                gt_paired_labels=valid_labels_list,
                pred_paired_labels=[torch.max(cls_scores_i, dim=-1)[1] for cls_scores_i in valid_cls_scores]
            )
            if num_classes == self.region_aware_branch.num_classes:
                self.matched_results['matched_result_r'] = last_matched_result
            elif num_classes == self.object_aware_branch.num_classes:
                self.matched_results['matched_result_o'] = last_matched_result
            else:
                self.matched_results.update(
                    last_matched_result
                )
        
        # concatenate all results from different samples
        cls_scores = torch.cat(valid_cls_scores) # [DT_NUM, C]
        labels = torch.cat(labels_list) # [DT_NUM]
        cls_weights = torch.cat(cls_weights) # [DT_NUM, C]
        gt_pts = torch.cat(gt_points_list) # [GT_NUM, 3]
        gt_paired_pts = torch.cat(gt_paired_pts) # [GT_NUM, 3]
        gt_pts_weights = torch.cat(gt_pts_weights) # [GT_NUM]
        pred_pts = torch.cat(valid_refine_pts) # [DT_NUM, 3]
        pred_paired_pts = torch.cat(pred_paired_pts) # [DT_NUM, 3]

        # calculate loss cls
        loss_cls = self.loss_cls(cls_scores,
                                labels,
                                weight=cls_weights,
                                avg_factor=cls_scores.shape[0])

        # calculate loss pts
        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts, 
                                  pred_paired_pts,
                                  avg_factor=pred_pts.shape[0])

        return loss_cls, loss_pts
    
    def loss_parts(self, all_cls_scores, all_refine_pts, voxel_semantics, mask_camera, target_classes, cls_weights):
        num_dec_layers = len(all_cls_scores)
        gt_points_list, gt_masks_list, gt_labels_list = \
            self.get_sparse_voxels(voxel_semantics, mask_camera, target_classes)
        
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_cls_weights_list = [cls_weights for _ in range(num_dec_layers)]
        
        losses_cls, losses_pts = multi_apply(
            self.loss_single, all_cls_scores, all_refine_pts, 
            all_gt_points_list, all_gt_masks_list, all_gt_labels_list, all_cls_weights_list)

        return losses_cls, losses_pts
        
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, mask_camera, preds_dicts):
        '''
            voxel_semantics: (B, H, W, D)
            mask_camera: (B, H, W, D)
            outs: dict{
                'init_points': (B, N, 1, 3)
                'all_cls_scores_r': [(B, N, 1, 6), (B, N, 16, 6), (B, N, 32, 6)]
                'all_refine_pts_r': [(B, N, 1, 3), (B, N, 16, 3), (B, N, 32, 3)]
                'all_cls_scores_o': [(B, N, 1, 11), (B, N, 2, 11), (B, N, 4, 11), (B, N, 8, 11), (B, N, 16, 11), (B, N, 32, 11)]
                'all_refine_pts_o': [(B, N, 1, 3), (B, N, 2, 3), (B, N, 4, 3), (B, N, 8, 3), (B, N, 16, 3), (B, N, 32, 3)]
            }
        '''
        loss_dict = dict()
        
        init_points = preds_dicts['init_points']
        all_cls_scores_r = preds_dicts['all_cls_scores_r'] # 3,B,N,K,6
        all_refine_pts_r = preds_dicts['all_refine_pts_r']
        target_classes_r = self.region_aware_branch.target_classes
        cls_weights_r = self.train_cfg.get('cls_weights_region', [1.0] * len(target_classes_r))
        
        all_cls_scores_o = preds_dicts['all_cls_scores_o'] # 6,B,N,K,11
        all_refine_pts_o = preds_dicts['all_refine_pts_o']
        target_classes_o = self.object_aware_branch.target_classes
        cls_weights_o = self.train_cfg.get('cls_weights_object', [1.0] * len(target_classes_o))
        
        # loss of all init_points
        if init_points is not None:
            gt_points_list, gt_masks_list, gt_labels_list = \
                self.get_sparse_voxels(voxel_semantics, mask_camera)
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)
            pseudo_cls_weights = [1.0 for _ in range(self.num_classes)]
            _, init_loss_pts = self.loss_single(
                pseudo_scores, init_points, 
                gt_points_list, gt_masks_list, gt_labels_list, pseudo_cls_weights)
            loss_dict['init_loss_pts'] = init_loss_pts

        # loss of region_aware_branch and object_aware_branch
        loss_cls_r, loss_pts_r = self.loss_parts(all_cls_scores_r, all_refine_pts_r, voxel_semantics, mask_camera, target_classes_r, cls_weights_r)
        loss_cls_o, loss_pts_o = self.loss_parts(all_cls_scores_o, all_refine_pts_o, voxel_semantics, mask_camera, target_classes_o, cls_weights_o)
        
        # loss from the last decoder layer
        loss_dict['loss_pts'] = loss_pts_r[-1] * self.branch_loss_weights[0] + loss_pts_o[-1] * self.branch_loss_weights[1]
        loss_dict['loss_cls'] = loss_cls_r[-1] * self.branch_loss_weights[2] + loss_cls_o[-1] * self.branch_loss_weights[3]

        # loss from other region_aware_branch decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(loss_cls_r[:-1], loss_pts_r[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls_region'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts_region'] = loss_pts_i
            num_dec_layer += 1
        
        # loss from other object_aware_branch decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(loss_cls_o[:-1], loss_pts_o[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls_object'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts_object'] = loss_pts_i
            num_dec_layer += 1
        return loss_dict
    
    def get_occ(self, pred_dicts, img_metas, rescale=False):
        all_refine_pts_r, all_refine_pts_o = pred_dicts['all_refine_pts_r'], pred_dicts['all_refine_pts_o']
        refine_pts = torch.cat([all_refine_pts_r[-1], all_refine_pts_o[-1]], dim=1)
        batch_size, Q, K, _ = refine_pts.shape
        all_cls_scores_r, all_cls_scores_o = pred_dicts['all_cls_scores_r'], pred_dicts['all_cls_scores_o']
        cls_scores_r, cls_scores_o = all_cls_scores_r[-1].sigmoid(), all_cls_scores_o[-1].sigmoid()
        Q1, Q2 = cls_scores_r.shape[1], cls_scores_o.shape[1]
        cls_scores = torch.zeros((batch_size, Q, K, self.num_classes), device=refine_pts.device)
        cls_scores[:, :Q1, :, self.region_aware_branch.target_classes] = cls_scores_r
        cls_scores[:, Q1:, :, self.object_aware_branch.target_classes] = cls_scores_o

        ctr_dist_thr = self.test_cfg.get('ctr_dist_thr', 3.)

        batch_size = refine_pts.shape[0]
        ctr_dist_thr = self.test_cfg.get('ctr_dist_thr', 3.)
        score_thr = self.test_cfg.get('score_thr', 0.)

        result_list = []
        
        for i in range(batch_size):
            refine_pts, cls_scores = refine_pts[i], cls_scores[i]
            refine_pts = decode_points(refine_pts, self.pc_range)

            # filter weak points by distance and score
            centers = refine_pts.mean(dim=1, keepdim=True)
            ctr_dists = torch.norm(refine_pts - centers, dim=-1)
            mask_dist = ctr_dists < ctr_dist_thr
            mask_score = (cls_scores > score_thr).any(dim=-1)
            mask = mask_dist & mask_score
            refine_pts = refine_pts[mask]
            cls_scores = cls_scores[mask]
            
            pts = torch.cat([refine_pts, cls_scores], dim=-1)
            pts_infos, voxels, num_pts = self.voxel_generator(pts)
            voxels = torch.flip(voxels, [1]).long()
            pts, scores = pts_infos[..., :3], pts_infos[..., 3:]
            scores = scores.sum(dim=1) / num_pts[..., None]

            if self.test_cfg.get('padding', True):
                occ = scores.new_zeros((self.voxel_num[0], self.voxel_num[1], 
                                        self.voxel_num[2], self.num_classes))
                occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = scores
                occ = occ.permute(3, 0, 1, 2).unsqueeze(0)
                # padding
                dilated_occ = F.max_pool3d(occ, 3, stride=1, padding=1)
                eroded_occ = -F.max_pool3d(-dilated_occ, 3, stride=1, padding=1)
                # repalce with original occ prediction
                original_mask = (occ > score_thr).any(dim=1, keepdim=True)
                original_mask = original_mask.expand_as(eroded_occ)
                eroded_occ[original_mask] = occ[original_mask]
                # sparse dense occ
                eroded_occ = eroded_occ.squeeze(0).permute(1, 2, 3, 0)
                voxels = torch.nonzero((eroded_occ > score_thr).any(dim=-1))
                scores = eroded_occ[voxels[:, 0], voxels[:, 1], voxels[:, 2], :]

            labels = scores.argmax(dim=-1)
            result_list.append(dict(
                sem_pred=labels.detach().cpu().numpy(),
                occ_loc=voxels.detach().cpu().numpy()))

        return result_list
    
    def get_sparse_voxels(self, voxel_semantics, mask_camera, target_classes=None):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()
        
        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, H, Z)
        coors = torch.stack([xx, yy, zz], dim=-1) # actual space

        gt_points_list, gt_masks_list, gt_labels_list = [], [], []
        
        for i in range(B):
            if target_classes is not None:
                mask = torch.isin(voxel_semantics[i], torch.tensor(target_classes, device=device))   
            else:
                mask = voxel_semantics[i] != self.empty_label            
                
            mask_voxel_semantics = voxel_semantics[i][mask]

            if target_classes is not None:
                max_class = max(target_classes)
                mapping = torch.arange(max_class + 1, device=device)
                for map_label, ori_label in enumerate(target_classes):
                    mapping[ori_label] = map_label
                mask_voxel_semantics = mapping[mask_voxel_semantics]
            gt_points_list.append(coors[mask])
            gt_masks_list.append(mask_camera[i][mask]) # camera mask and not empty
            gt_labels_list.append(mask_voxel_semantics)

        return gt_points_list, gt_masks_list, gt_labels_list
