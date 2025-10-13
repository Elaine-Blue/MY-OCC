import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops import knn, Voxelization
from mmdet.core import multi_apply
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from ..bbox.utils import decode_points
import pickle
import copy

@HEADS.register_module()
class ObjectAwareHead(BaseModule):
    '''
        Object-Aware Branch to predict car, truck, bus, pedestrian etc dynamics.
    '''
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 transformer=None,
                 pc_range=[],
                 empty_label=17,
                 voxel_size=[],
                 target_classes=[],
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.empty_label = empty_label
        self.target_classes = target_classes
        self.transformer = build_transformer(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims
        self.voxel_generator = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points=10, 
            max_voxels=self.num_query * self.num_refines[-1],
            deterministic=False
        )
        
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()
        
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)

        self._init_layers()
        
    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0, 1)

    def init_weights(self):
        self.transformer.init_weights()

    def obtain_proposals(self, img, img_metas, training_mode=True):
        self.eval()
        detection_h, detection_w = 736, 1280
        camera_h, camera_w = 900, 1600
        bs, num_cameras, channels, hs, ws = img.shape
        batch_results_proposals = []
        for bs in range(len(img_metas)):
            results_proposals = []
            cam_intrinsics = img_metas[bs]['intrinsics'].data
            cameras_intrinsics = img_metas[bs]['proposals_intrinsics']
            cam_results = img_metas[bs]['results_plane']
            cam_lidar2cams = img_metas[bs]['extrinsics'].data
            for cam_idx in range(img.size(1)):
                if False:
                    img_path = img_metas[bs]['filename'][cam_idx]
                    img_name = img_path.split('/')[-1]
                    cam_image = Image.open(img_path)
                    resized_image = transforms.Resize(
                        size=(camera_h, camera_w))(cam_image)
                    cam_sample = normalize(resized_image)
                    cam_sample = cam_sample[None].to(img.device)

                cam_intrinsic = cameras_intrinsics[cam_idx]
                cam_results_2d = cam_results[cam_idx]
                cam_cam2lidar = np.linalg.inv(cam_lidar2cams[cam_idx])
                results_proposals_cam = []
                # bbox2ds = []
                for ob_idx, result_2d in enumerate(cam_results_2d):
                    bbox2d, class_idx, score, _ = result_2d
                    bbox2d = torch.cat(
                        [torch.tensor([ob_idx]), torch.from_numpy(bbox2d)], dim=0).to(img.device)

                    # 2d results on (900, 1600)
                    scale_factors = torch.zeros_like(bbox2d[1:])
                    scale_factors[0::2] = camera_w / detection_w
                    scale_factors[1::2] = camera_h / detection_h
                    bbox2d[1:] = scale_factors * bbox2d[1:]
                    class_idx = torch.tensor(class_idx).to(img.device)

                    init_proposals_corners, proposals_bboxs, object_class, ious = init_proposal_anchors(
                        class_idx, bbox2d[1:], cam_intrinsic, cam_cam2lidar)
                    if ious.size(0) >= 1:
                        # nms for redunant 3d proposals
                        _, max_ious_idx = torch.topk(
                            ious, k=min(3, ious.size(0)), dim=0, largest=True)
                        proposals_bboxs = proposals_bboxs[max_ious_idx]
                        if False:  # check
                            initial_proposals = init_proposals_corners[max_ious_idx]
                            check_centers = initial_proposals.mean(1)
                            proposals_bboxs = proposals_bboxs
                            centers_embed = proposals_bboxs[..., :3]
                            sizes_embed = proposals_bboxs[..., 3:6]
                            thetas_ = proposals_bboxs[..., -1:]
                            for s, t, c, pt_3d in zip(sizes_embed, thetas_, centers_embed, initial_proposals):
                                w, h, l = s
                                corner_3d = check_3d(c, w, h, l, t)
                                print(1)
                        proposals_bboxs = normlize_boxes(
                            proposals_bboxs, cam_cam2lidar)
                        results_proposals_cam += [(bbox2d, object_class, p)
                                                  for p in proposals_bboxs]
                    # bbox2ds.append(bbox2d[1:])
                results_proposals.append(results_proposals_cam)
            batch_results_proposals.append(results_proposals)
        if training_mode:
            self.train()
        return batch_results_proposals

    def roi_pooling(self, feats, rois, original_size, size=1):
        cf, hf, wf = feats.size()
        assert rois.size(1) == 5
        output = []
        rois = rois.data.float()
        num_rois = rois.size(0)
        norm_oris = rois[:, 1:] / torch.tensor([original_size[0], original_size[1],
                                               original_size[0], original_size[1]])[None].repeat(num_rois, 1).to(feats.device)
        rois = torch.cat(
            [rois[:, :1], norm_oris * torch.tensor([wf, hf, wf, hf])[None].repeat(num_rois, 1).to(feats.device)], dim=1)
        rois = rois.long()
        for i in range(num_rois):
            roi = rois[i]
            im_idx = roi[0]
            im = feats[..., roi[2].clamp(min=0):(
                roi[4].clamp(max=hf) + 1), roi[1].clamp(min=0):(roi[3].clamp(max=wf) + 1)]
            output.append(F.adaptive_max_pool2d(im, size))

        output = torch.stack(output, 0)
        return output.squeeze(-1).squeeze(-1)

    def extract_roi_feats(self, img_feats, init_proposals):
        level_roi_embed = []

        for num_level in range(len(img_feats)):
            level_feats = img_feats[num_level]
            batch_roi_embed = []
            for b in range(level_feats.size(0)):
                cam_roi_embed = []
                for cam in range(level_feats.size(1)):
                    cam_proposal_b = [bbox2ds[0]
                                      for bbox2ds in init_proposals[b][cam]]
                    if len(cam_proposal_b) > 0:
                        cam_proposal_b = torch.stack(cam_proposal_b, dim=0)
                        cam_roi_embed += [self.roi_pooling(
                            level_feats[b, cam], cam_proposal_b, (1600, 900))]   # (1280, 736)
                    else:
                        cam_roi_embed += []
                batch_roi_embed += [cam_roi_embed]
            level_roi_embed += batch_roi_embed  # batch, cam, objects, dim

        return level_roi_embed
    
    def forward(self, mlvl_feats, img_metas):
        B, Q, = mlvl_feats[0].shape[0], self.num_query
        init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1)
        query_feat = init_points.new_zeros(B, Q, self.embed_dims)

        cls_scores, refine_pts = self.transformer(
            init_points,
            query_feat,
            mlvl_feats,
            img_metas=img_metas,
        )
        return init_points, cls_scores, refine_pts
