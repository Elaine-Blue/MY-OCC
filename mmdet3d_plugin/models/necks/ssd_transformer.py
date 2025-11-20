# import sys
# sys.path.append('/root/autodl-tmp/Project/MY-OCC/mmdet3d_plugin/models')
# sys.path.append('/root/autodl-tmp/Project/MY-OCC/mmdet3d_plugin/models/necks')
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import bias_init_with_prob, Scale
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmcv.ops import knn
from mmdet.models.utils.builder import TRANSFORMER
from ..bbox.utils import decode_bbox, decode_points, encode_points
from ..utils import inverse_sigmoid, DUMP
from ..occ_sampling import sampling_4d
from ..checkpoint import checkpoint as cp
from ..csrc.wrapper import MSMV_CUDA

@TRANSFORMER.register_module()
class SSDTransformerDecoder(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_refines=16,
                 num_groups=4,
                 scales=[1.0],
                 pc_range=[],
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_groups = num_groups

        if len(scales) == 1:
            scales = scales * num_layers
        if not isinstance(num_refines, list):
            num_refines = [num_refines]
        if len(num_refines) == 1:
            num_refines = num_refines * num_layers
            
        self.num_refines = num_refines
        last_refines = [1] + num_refines

        # params are shared across all decoder layers
        self.decoder_layers = ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                SSDTransformerDecoderLayer(
                    embed_dims, num_frames, num_views, num_points, num_levels, num_classes, 
                    num_groups, num_refines[i], last_refines[i], layer_idx=i, 
                    scale=scales[i], pc_range=pc_range)
            )
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        self.decoder_layers.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, img_metas):
        cls_scores, refine_pts = [], []
        ego2img = np.asarray([m['ego2img'] for m in img_metas]).astype(np.float32)
        ego2img = query_feat.new_tensor(ego2img) # [B, N, 4, 4]
        ego2occ = np.asarray([m['ego2occ'] for m in img_metas]).astype(np.float32)
        occ2ego = torch.inverse(query_feat.new_tensor(ego2occ))
        occ2ego = occ2ego[:, None].expand_as(ego2img)
        occ2img = ego2img @ occ2ego

        # group image features in advance for sampling, see `sampling_4d` for more details
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
            N, T, G, C = self.num_views, self.num_frames, self.num_groups, GC//self.num_groups
            assert T*N == TN
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:  # Our CUDA operator requires channel_last
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
                feat = feat.reshape(B*T*G, N, H, W, C)
            else:  # Torch's grid_sample requires channel_first
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)  # [B, T, G, C, N, H, W]
                feat = feat.reshape(B*T*G, C, N, H, W)

            mlvl_feats[lvl] = feat.contiguous()

        for i, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i

            # TODO: detach() or not is better to be studied
            query_points = query_points.detach()
            query_feat, cls_score, query_points = decoder_layer(
                query_points, query_feat, mlvl_feats, occ2img, img_metas)

            cls_scores.append(cls_score)
            refine_pts.append(query_points)

        
        cls_scores = [torch.nan_to_num(score) for score in cls_scores]
        refine_pts = [torch.nan_to_num(pts) for pts in refine_pts]

        return cls_scores, refine_pts


class SSDTransformerDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=16,
                 last_refines=16,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 layer_idx=0,
                 scale=1.0,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.num_points = num_points
        self.num_refines = num_refines
        self.last_refines = last_refines
        self.layer_idx = layer_idx
        self.scale = scale

        self.position_encoder = nn.Sequential(
            nn.Linear(3 * self.last_refines, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = SSDSelfAttention(
            embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = SSDSampling(embed_dims, num_frames=num_frames, num_views=num_views,
                                     num_groups=num_groups, num_points=num_points, 
                                     num_levels=num_levels, pc_range=pc_range)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames,
                                     n_groups=num_groups, out_points=32)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(
            self.embed_dims, self.num_classes * self.num_refines))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, 3 * self.num_refines))
        self.reg_branch = nn.Sequential(*reg_branch)

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def refine_points(self, points_proposal, points_delta):
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, self.num_refines, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    def forward(self, query_points, query_feat, mlvl_feats, occ2img, img_metas):
        """
        query_points: [B, Q, 3] [x, y, z]
        """
        query_pos = self.position_encoder(query_points.flatten(2, 3))
        query_feat = query_feat + query_pos
        
        sampled_feat = self.sampling(
            query_points, query_feat, mlvl_feats, occ2img, img_metas) #[B, Q, G, FP, C]
        query_feat = self.norm1(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm2(self.self_attn(query_points, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))

        B, Q = query_points.shape[:2]
        cls_score = self.cls_branch(query_feat)  # [B, Q, P * num_classes]
        reg_offset = self.scale * self.reg_branch(query_feat)  # [B, Q, P * 3]
        cls_score = cls_score.reshape(B, Q, self.num_refines, self.num_classes)
        refine_pt = self.refine_points(query_points, reg_offset)
        return query_feat, cls_score, refine_pt


class SSDSelfAttention(BaseModule):
    """Scale-adaptive Self Attention"""
    def __init__(self, 
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = pc_range

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_points, query_feat):
        """
        query_points: [B, Q, 3]
        query_feat: [B, Q, C]
        """
        dist = self.calc_points_dists(query_points)
        tau = self.gen_tau(query_feat)  # [B, Q, 8]

        if DUMP.enabled:
            torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_points, query_feat):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat,
                      use_reentrant=False)
        else:
            return self.inner_forward(query_points, query_feat)

    @torch.no_grad()
    def calc_points_dists(self, points):
        points = decode_points(points, self.pc_range)
        if points.shape[2] != 3:
            points = points.mean(dim=2)
        dist = torch.norm(points.unsqueeze(-2) - points.unsqueeze(-3), dim=-1)
        return -dist


class SSDSampling(BaseModule):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self,
                 embed_dims=256,
                 num_frames=4,
                 num_views=6,
                 num_groups=4,
                 num_points=8,
                 num_levels=4,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_views = num_views
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_points, query_feat, mlvl_feats, occ2img, img_metas):
        '''
        query_points: [B, Q, 6]
        query_feat: [B, Q, C]
        '''
        B, Q = query_points.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # query points
        query_points = decode_points(query_points, self.pc_range)
        if query_points.shape[2] == 1:
            query_center = query_points
            query_scale = torch.zeros_like(query_center)
        elif query_points.shape[2] == 3:
            query_center = query_points.unsqueeze(2)
            query_scale = torch.ones_like(query_center)
        else:
            query_center = query_points.mean(dim=2, keepdim=True)
            query_scale = query_points.std(dim=2, keepdim=True)

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, -1, 3)

        sampling_points = query_center + sampling_offset * query_scale
        sampling_points = sampling_points.view(B, Q, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            occ2img,
            image_h, image_w,
            self.num_views
        )  # [B, Q, G, FP, C]

        return sampled_feats

    def forward(self, query_points, query_feat, mlvl_feats, occ2img, img_metas):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat, mlvl_feats,
                      occ2img, img_metas, use_reentrant=False)
        else:
            return self.inner_forward(query_points, query_feat, mlvl_feats,
                                      occ2img, img_metas)


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super().__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        out = x.reshape(B*Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B*Q, G, self.out_points, self.in_points)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)


@TRANSFORMER.register_module()
class BoundaryTransformerDecoder(BaseModule):
    """
    Boundary-aware transformer decoder with distance prediction.
    
    Extends SSDTransformerDecoder to predict boundary distances in addition
    to semantic classes.
    """
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_directions=6,
                 num_groups=4,
                 scales=[1.0],
                 pc_range=[],
                 occupancy_prior_path=None,
                 grid_shape=(200, 200, 16),
                 use_occupancy_prior=True,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_groups = num_groups
        self.num_directions = num_directions
        self.num_refines = num_directions + 1

        if len(scales) == 1:
            scales = scales * num_layers

        # Decoder layers with boundary prediction
        self.decoder_layers = ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                BoundaryTransformerDecoderLayer(
                    embed_dims, num_frames, num_views, num_points, num_levels,
                    num_classes, num_directions, num_groups, layer_idx=i, scale=scales[i],
                    pc_range=pc_range,
                    occupancy_prior_path=occupancy_prior_path,
                    grid_shape=grid_shape,
                    use_occupancy_prior=use_occupancy_prior)
            )
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        self.decoder_layers.init_weights()
    
    def forward(self, query_points, query_feat, mlvl_feats, img_metas, 
                gt_bboxes_3d=None, gt_labels_3d=None):
        """
        Forward pass with boundary distance prediction.
        
        Returns:
            cls_scores: List of (B, N, P, C) classification scores per layer
            refine_pts: List of (B, N, P, 3) refined points per layer
            boundary_dists: List of (B, N, P, K) boundary distances per layer
        """
        cls_scores, refine_pts, boundary_dists = [], [], []
        
        # Prepare projection matrices
        ego2img = np.asarray([m['ego2img'] for m in img_metas]).astype(np.float32)
        ego2img = query_feat.new_tensor(ego2img)  # [B, N, 4, 4]
        ego2occ = np.asarray([m['ego2occ'] for m in img_metas]).astype(np.float32)
        occ2ego = torch.inverse(query_feat.new_tensor(ego2occ))
        occ2ego = occ2ego[:, None].expand_as(ego2img)
        occ2img = ego2img @ occ2ego
        
        # Group image features for sampling
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape
            N, T, G, C = self.num_views, self.num_frames, self.num_groups, GC//self.num_groups
            assert T*N == TN
            feat = feat.reshape(B, T, N, G, C, H, W)
            
            if MSMV_CUDA:
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
                feat = feat.reshape(B*T*G, N, H, W, C)
            else:
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)  # [B, T, G, C, N, H, W]
                feat = feat.reshape(B*T*G, C, N, H, W)
            
            mlvl_feats[lvl] = feat.contiguous()
        
        for i, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i
            
            query_points = query_points.detach()
            (query_feat, cls_score, query_points, boundary_dist) =\
                decoder_layer(query_points, query_feat, mlvl_feats, occ2img, img_metas)
            
            cls_scores.append(cls_score)
            refine_pts.append(query_points)
            boundary_dists.append(boundary_dist)
        
        # Handle NaN values
        cls_scores = [torch.nan_to_num(score) for score in cls_scores]
        refine_pts = [torch.nan_to_num(pts) for pts in refine_pts]
        boundary_dists = [torch.nan_to_num(dist) for dist in boundary_dists]
        
        return cls_scores, refine_pts, boundary_dists


class BoundaryTransformerDecoderLayer(BaseModule):
    """
    Single decoder layer with boundary distance prediction.
    
    Predicts:
    1. Semantic class (num_classes)
    2. Point refinement offsets (3D)
    3. Boundary distances in K directions
    """
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_levels=4,
                 num_classes=10,
                 num_directions=6,
                 num_groups=4,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 layer_idx=0,
                 scale=1.0,
                 pc_range=[],
                 occupancy_prior_path=None,
                 grid_shape=(200, 200, 16),
                 use_occupancy_prior=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_directions = num_directions
        self.num_refines = num_directions + 1
        self.pc_range = pc_range
        self.num_points = num_points
        self.layer_idx = layer_idx
        self.scale = scale
        self.use_occupancy_prior = use_occupancy_prior

        # Position encoder - use occupancy-aware encoder if enabled
        if use_occupancy_prior:
            self.position_encoder = OccupancyPriorPositionEncoder(
                embed_dims=embed_dims,
                prior_path=occupancy_prior_path,
                num_freq_bands=8,
                grid_shape=grid_shape,
                pc_range=pc_range,
                use_trilinear=True
            )
        else:
            # Fallback to simple MLP
            self.position_encoder = nn.Sequential(
                nn.Linear(3, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )
        
        # boundary offsets
        self.boundary_offsets = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.num_directions),
        )

        # Attention modules
        self.self_attn = SSDSelfAttention(
            embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = SSDSampling(
            embed_dims, num_frames=num_frames, num_views=num_views,
            num_groups=num_groups, num_points=num_points,
            num_levels=num_levels, pc_range=pc_range)
        self.mixing = AdaptiveMixing(
            in_dim=embed_dims, in_points=num_points * num_frames,
            n_groups=num_groups, out_points=32)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        
        # Classification branch
        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(
            self.embed_dims, self.num_refines * self.num_classes))
        self.cls_branch = nn.Sequential(*cls_branch)
        
        # Regression branch (point refinement)
        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, self.num_refines * 3))
        self.reg_branch = nn.Sequential(*reg_branch)
    
    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()
        
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)
    
    def refine_points(self, points_proposal, points_delta):
        """Refine point positions."""
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, 3)
        
        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)
    
    def create_boundary_points(self, query_points, boundary_offsets):
        """Create boundary points."""
        query_points = query_points.repeat(1, 1, self.num_directions + 1, 1)
        query_points[:, :, 1, 0] += boundary_offsets[..., 0] # +x
        query_points[:, :, 2, 0] += boundary_offsets[..., 1] # -x
        query_points[:, :, 3, 1] += boundary_offsets[..., 2] # +y
        query_points[:, :, 4, 1] += boundary_offsets[..., 3] # -y
        query_points[:, :, 5, 2] += boundary_offsets[..., 4] # +z
        query_points[:, :, 6, 2] += boundary_offsets[..., 5] # -zs
        return query_points
    
    def forward(self, query_points, query_feat, mlvl_feats, occ2img, img_metas):
        """
        Forward pass with boundary prediction.
        
        Args:
            query_points: (B, Q, K, 3) query positions
            query_feat: (B, Q, K, C) query features
            mlvl_feats: Multi-level image features
            occ2img: Projection matrices
            img_metas: Image meta information
        
        Returns:
            query_feat: (B, Q, K, C) updated query features
            cls_score: (B, Q, K, C) classification scores
            refine_pts: (B, Q, P, 3) refined point positions
            boundary_dist: (B, Q, P, K) boundary distances
            occupancy_conf: (B, Q, P) occupancy confidence
        """
        B, Q = query_points.shape[:2]
        
        # Add six-dimensional boundary points
        boundary_offsets = self.boundary_offsets(query_feat)
        query_points = self.create_boundary_points(query_points, boundary_offsets)

        # Position encoding
        query_feat = query_feat.repeat(1, 1, self.num_refines, 1)
        query_pos = self.position_encoder(query_points.flatten(1, 2))
        query_feat = query_feat + query_pos.reshape(B, Q, self.num_refines, -1)
        
        # Cross-attention: sample from multi-view features
        query_points = query_points.flatten(1, 2)
        query_feat = query_feat.flatten(1, 2)
        sampled_feat = self.sampling(
            query_points, query_feat, mlvl_feats, occ2img, img_metas)
        query_feat = self.norm1(self.mixing(sampled_feat, query_feat))
        
        # Self-attention
        query_feat = self.norm2(self.self_attn(query_points, query_feat))
        
        # Feed-forward network
        query_feat = self.norm3(self.ffn(query_feat))
        
        # Predictions
        cls_score = self.cls_branch(query_feat)  # (B, Q, P*C)
        cls_score = cls_score.reshape(B, Q, self.num_refines, self.num_classes)
        
        reg_offset = self.scale * self.reg_branch(query_feat)  # (B, Q, P*3)
        refine_pt = self.refine_points(query_points, reg_offset)

        return query_feat, cls_score, refine_pt, boundary_offsets
