import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import bias_init_with_prob, Scale
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmdet.models.utils.builder import TRANSFORMER
from ..bbox.utils import decode_bbox, decode_points, encode_points
from ..utils import inverse_sigmoid, DUMP
from ..occ_sampling import sampling_4d
from ..checkpoint import checkpoint as cp
from ..csrc.wrapper import MSMV_CUDA
from scipy.interpolate import RegularGridInterpolator
import os, pickle


class PositionEncoder(BaseModule):
    """
    Fourier Position Encoder with Occupancy Probability Weighting

    For each point (x, y, z), computes:
    1. Relative distance to scene center for each axis
    2. Fourier encoding for each axis separately
    3. Sum the three-axis encodings
    4. Multiply by occupancy probability at that location
    """
    # Class-level cache for shared occupancy interpolator
    _occ_interpolator_cache = {}

    def __init__(self,
                 embed_dims=256,
                 grid_size=(200, 200, 16),  # meters
                 occ_prior_path=None,
                 pc_range=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_freqs = embed_dims // 2
        self.grid_size = grid_size
        self.scene_size = (pc_range[3] - pc_range[0],
                           pc_range[4] - pc_range[1],
                           pc_range[5] - pc_range[2])  # in meters
        self.pc_range = pc_range
        self.scene_center = torch.tensor([0.0, 0.0, 0.0])

        '''
            Fourier encoding:
                for each axis, encode as [sin(2^0*pi*d), cos(2^0*pi*d), ..., sin(2^(L-1)*pi*d), cos(2^(L-1)*pi*d)]
                total encoding dim per axis = 2 * num_freqs, total encoding dim = 3 * 2 * num_freqs
        '''
        self.fourier_dim = 2 * self.num_freqs

        # Project Fourier encoding to embed_dims
        self.pos_proj = nn.Sequential(
            nn.Linear(self.fourier_dim, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

        # Load or retrieve occupancy prior interpolator from cache
        self.occ_interpolator = None
        if occ_prior_path is not None and os.path.exists(occ_prior_path):
            # Check if already loaded in cache
            if occ_prior_path in PositionEncoder._occ_interpolator_cache:
                self.occ_interpolator = PositionEncoder._occ_interpolator_cache[occ_prior_path]
            else:
                # Load and cache the interpolator
                try:
                    with open(occ_prior_path, 'rb') as f:
                        occ_stats = pickle.load(f)
                    occupancy_density = occ_stats['occupancy_density']

                    # Create interpolator for occupancy probability
                    X_dim, Y_dim, Z_dim = occupancy_density.shape
                    self.occ_interpolator = RegularGridInterpolator(
                        (np.arange(X_dim), np.arange(Y_dim), np.arange(Z_dim)),
                        occupancy_density,
                        method='linear',
                        bounds_error=False,
                        fill_value=0.5  # Default occupancy probability
                    )
                    # Cache for future instances
                    PositionEncoder._occ_interpolator_cache[occ_prior_path] = self.occ_interpolator
                    print(f"Loaded and cached occupancy prior from {occ_prior_path}")
                except Exception as e:
                    print(f"Warning: Failed to load occupancy prior: {e}")
                    self.occ_interpolator = None

        # Frequency bands for Fourier encoding
        freq_bands = 2.0 * torch.linspace(0, self.num_freqs - 1, self.num_freqs)
        self.register_buffer('freq_bands', freq_bands)

    def get_rel_dist(self, points):
        """
        Calculate relative distance to scene center for each axis.

        Args:
            points: [B, Q, K, 3] in world coordinates

        Returns:
            relative_dist: [B, Q, K, 3] normalized to [-1, 1]
        """
        device = points.device
        scene_center = self.scene_center.to(device)
        dist_to_center = points - scene_center.view(1, 1, 1, 3)
        scene_size = torch.tensor(self.scene_size, device=device).view(1, 1, 1, 3)
        rel_dist = dist_to_center / (scene_size / 2.0)
        # Clamp to [-1, 1]
        rel_dist = torch.clamp(rel_dist, -1.0, 1.0)
        return rel_dist

    def fourier_encode(self, rel_dist):
        """
        Apply Fourier encoding to relative distances.

        Args:
            rel_dist: [B, Q, K, 3] normalized distances

        Returns:
            encoding: [B, Q, K, fourier_dim]
        """
        B, Q, K, _ = rel_dist.shape

        # Split into x, y, z components
        x_dist = rel_dist[..., 0:1]  # [B, Q, K, 1]
        y_dist = rel_dist[..., 1:2]  # [B, Q, K, 1]
        z_dist = rel_dist[..., 2:3]  # [B, Q, K, 1]

        encodings = []

        # Encode each axis separately
        for dist in [x_dist, y_dist, z_dist]:
            freq_dist = dist * self.freq_bands.view(1, 1, 1, -1) * np.pi  # [B, Q, K, num_freqs]

            sin_encoding = torch.sin(freq_dist)  # [B, Q, K, num_freqs]
            cos_encoding = torch.cos(freq_dist)  # [B, Q, K, num_freqs]

            # Interleave sin and cos: [sin(f0), cos(f0), sin(f1), cos(f1), ...]
            axis_encoding = torch.stack([sin_encoding, cos_encoding], dim=-1)  # [B, Q, K, num_freqs, 2]
            axis_encoding = axis_encoding.reshape(B, Q, K, -1)  # [B, Q, K, 2*num_freqs]

            encodings.append(axis_encoding)

        fourier_encoding = encodings[0] + encodings[1] + encodings[2]
        return fourier_encoding

    def get_occ_prob(self, points):
        """
        Query occupancy probability at given points.

        Args:
            points: [B, Q, K, 3] in world coordinates

        Returns:
            occ_prob: [B, Q, K, 1]
        """
        if self.occ_interpolator is None:
            return torch.ones_like(points[..., 0:1])

        B, Q, K, _ = points.shape
        device = points.device

        if len(self.pc_range) >= 6:
            # Map from world coords to grid coords [0, grid_dim)
            x_min, y_min, z_min = self.pc_range[0], self.pc_range[1], self.pc_range[2]
            x_max, y_max, z_max = self.pc_range[3], self.pc_range[4], self.pc_range[5]

            # Normalize to [0, 1]
            x_norm = (points[..., 0] - x_min) / (x_max - x_min)
            y_norm = (points[..., 1] - y_min) / (y_max - y_min)
            z_norm = (points[..., 2] - z_min) / (z_max - z_min)

            grid_coords = torch.stack([
                x_norm * (self.grid_size[0] - 1),
                y_norm * (self.grid_size[1] - 1),
                z_norm * (self.grid_size[2] - 1)
            ], dim=-1)
            
            grid_coords_np = grid_coords.detach().cpu().numpy().reshape(-1, 3)
            occ_probs_np = self.occ_interpolator(grid_coords_np)
            occ_probs = torch.tensor(occ_probs_np, device=device, dtype=points.dtype)
            occ_probs = occ_probs.reshape(B, Q, K, 1)
        else:
            occ_probs = torch.ones(B, Q, K, 1, device=device, dtype=points.dtype)

        return occ_probs

    def forward(self, query_points):
        """
        Args:
            query_points: [B, Q, K, 3] in encoded/normalized coordinates

        Returns:
            pos_encoding: [B, Q, K, embed_dims]
        """
        points_ego = decode_points(query_points, self.pc_range)
        rel_dist = self.get_rel_dist(points_ego)
        fourier_features = self.fourier_encode(rel_dist)  # [B, Q, K, fourier_dim]
        pos_encoding = self.pos_proj(fourier_features)  # [B, Q, K, embed_dims]
        occ_prob = self.get_occ_prob(points_ego)  # [B, Q, K, 1]
        pos_encoding = pos_encoding * occ_prob  # [B, Q, K, embed_dims]
        return pos_encoding

@TRANSFORMER.register_module()
class OccTransformer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=[1, 2, 4, 8, 16, 32],
                 scales=[1.0],
                 pc_range=[],
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_refines = num_refines
        
        self.decoder = TransformerDecoder(
            embed_dims, num_frames, num_views, num_points, num_layers, num_levels,
            num_classes, num_refines, num_groups, scales, pc_range)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, img_metas):
        cls_scores, refine_pts = self.decoder(
            query_points, query_feat, mlvl_feats, img_metas)

        cls_scores = [torch.nan_to_num(score) for score in cls_scores]
        refine_pts = [torch.nan_to_num(pts) for pts in refine_pts]

        return cls_scores, refine_pts


class TransformerDecoder(BaseModule):
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
        super().__init__(init_cfg)
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
        last_refines = [1] + num_refines

        # params are shared across all decoder layers
        self.decoder_layers = ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                TransformerDecoderLayer(
                    embed_dims, num_frames, num_views, num_points, num_levels, num_classes, 
                    num_groups, num_refines[i], last_refines[i], layer_idx=i, 
                    scale=scales[i], pc_range=pc_range)
            )
        
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

            query_points = query_points.detach()
            query_feat, cls_score, query_points = decoder_layer(
                query_points, query_feat, mlvl_feats, occ2img, img_metas)

            cls_scores.append(cls_score)
            refine_pts.append(query_points)

        return cls_scores, refine_pts

class TransformerDecoderLayer(BaseModule):
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
                 occ_prior_path='data/nuscenes/occupancy_prior.pkl',
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

        # Use new Fourier-based position encoder with occupancy prior
        self.position_encoder = PositionEncoder(
            embed_dims=embed_dims,
            grid_size=(200, 200, 16),
            occ_prior_path=occ_prior_path,
            pc_range=pc_range
        )

        self.self_attn = SelfAttention(
            embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = Sampling(embed_dims, num_frames=num_frames, num_views=num_views,
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
        query_points: [B, Q, K, 3] [x, y, z]
        """
        # Use new position encoder that outputs [B, Q, K, embed_dims]
        query_pos = self.position_encoder(query_points)  # [B, Q, K, embed_dims]

        # Average over K dimension to get [B, Q, embed_dims]
        query_pos = query_pos.mean(dim=2)  # [B, Q, embed_dims]

        query_feat = query_feat + query_pos

        sampled_feat = self.sampling(
            query_points, query_feat, mlvl_feats, occ2img, img_metas)
        query_feat = self.norm1(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm2(self.self_attn(query_points, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))

        B, Q = query_points.shape[:2]
        cls_score = self.cls_branch(query_feat)  # [B, Q, P * num_classes]
        reg_offset = self.scale * self.reg_branch(query_feat)  # [B, Q, P * 3]
        cls_score = cls_score.reshape(B, Q, self.num_refines, self.num_classes)
        refine_pt = self.refine_points(query_points, reg_offset)

        if DUMP.enabled:
            pass # TODO: enable OTR dump

        return query_feat, cls_score, refine_pt


class SelfAttention(BaseModule):
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
        query_points: [B, Q, K, 3]
        query_feat: [B, Q, C]
        """
        dist = self.calc_points_dists(query_points) # [B, Q, Q]
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
        points = points.mean(dim=2)
        dist = torch.norm(points.unsqueeze(-2) - points.unsqueeze(-3), dim=-1)
        return -dist


class Sampling(BaseModule):
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
        query_points: [B, Q, K, 3]
        query_feat: [B, Q, C]
        '''
        B, Q = query_points.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # query points
        query_points = decode_points(query_points, self.pc_range)
        if query_points.shape[2] == 1:
            query_center = query_points
            query_scale = torch.zeros_like(query_center)
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


class SSDAttention(BaseModule):
    """
    Function: 
        Geometry-Guided Self Attention:
        1. Compute soft geometry mask based on point-box affinity
        2. Integrate geometry mask into self-attention
    """
    def __init__(self, 
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 pc_range=[],
                 init_cfg=None,
                 same_box_weight=2.0,
                 same_class_weight=1.5,
                 distance_threshold=1.0):
        super().__init__(init_cfg)
        self.pc_range = pc_range
        self.same_box_weight = same_box_weight
        self.same_class_weight = same_class_weight
        self.distance_threshold = distance_threshold
        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)
        
    def compute_attn_mask(self, query_points, gt_bboxes_3d, gt_labels_3d, spatial_dists):
        B, Q, _ = query_points.shape
        device = query_points.device
        
        attn_mask = torch.zeros(B, Q, Q, device=device)
        
        for batch_idx in range(B):
            batch_points = query_points[batch_idx]
            batch_boxes = gt_bboxes_3d[batch_idx].tensor.to(device)  # [N, 9]
            
            if len(batch_boxes) == 0:
                continue
                
            box_probs, in_box_mask = self.compute_point_box_affinity(batch_points, batch_boxes)  # [Q, N]
            point_similarity = torch.matmul(box_probs, box_probs.T)  # [Q, Q]

            spatial_dist = spatial_dists[batch_idx] / (self.pc_range[3] - self.pc_range[0])  # [Q, Q]
            distance_decay = torch.exp(1 + spatial_dist)
            
            geometry_mask = point_similarity * distance_decay
            bbox_indices = torch.argmax(box_probs, dim=1)  # [Q]
            query_classes = gt_labels_3d[batch_idx][bbox_indices]  # [Q]
            class_mask = query_classes.unsqueeze(0) == query_classes.unsqueeze(1)  # [Q, Q]
            in_box_mask = in_box_mask.any(dim=1)  # [Q]
            box_mask = in_box_mask.unsqueeze(0) & in_box_mask.unsqueeze(1)  # [Q, Q]
            class_mask[~box_mask] = False
            class_mask = class_mask.float()
            attn_mask[batch_idx] = geometry_mask * self.same_box_weight + class_mask * self.same_class_weight
        return attn_mask
        
    
    def compute_point_box_affinity(self, points, boxes_3d):
        Q = points.shape[0]
        N = boxes_3d.shape[0]
        
        affinities = torch.zeros(Q, N, device=points.device)
        in_box_mask = torch.zeros(Q, N, dtype=bool, device=points.device)
        
        for box_idx in range(N):
            box = boxes_3d[box_idx]
            center, size, heading = box[:3], box[3:6], box[6]
            
            local_points = self.transform_to_box_coordinates(points, center, heading)
            
            center_dist = torch.norm(local_points, dim=1)
            
            inside_prob = torch.sigmoid((size.norm() / 2 - center_dist) * 5.0)
            affinities[:, box_idx] = inside_prob
            in_box_mask[:, box_idx] = (center_dist <= (size.norm() / 2 + 0.1))
        
        affinities = torch.softmax(affinities * 10, dim=1)
        return affinities, in_box_mask

    def transform_to_box_coordinates(self, points, center, heading):
        """
        Convert global points to local box coordinates.
        
        Args:
            points: [Q, 3]
            center: [3]
            heading: [1]
        
        Returns:
            local_points: [Q, 3]
        """
        device = points.device
        Q = points.shape[0]
        
        translated = points - center.unsqueeze(0)  # [Q, 3]

        cos_theta = torch.cos(-heading)
        sin_theta = torch.sin(-heading)
        
        rot_matrix = torch.tensor([
            [cos_theta, -sin_theta, 0],
            [sin_theta,  cos_theta, 0],
            [0,          0,         1]
        ], device=device, dtype=points.dtype)  # [3, 3]
        
        local_points = torch.matmul(translated, rot_matrix.T)  # [Q, 3]
    
        return local_points
    
    def inner_forward(self, query_points, query_feat, gt_bboxes_3d, gt_labels_3d):
        '''
            Args:
                query_points: [B, Q, K, 3]
                query_feat: [B, Q, C]
                gt_bboxes_3d: list of [LiDARInstance3DBoxes(N, 9)] * B
                gt_labels_3d: list of [N] * B
        '''
        dist = self.calc_points_dists(query_points)
        tau = self.gen_tau(query_feat)
        
        if DUMP.enabled:
            torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        query_points = query_points.mean(dim=2)  # [B, Q, 3]
        query_points = decode_points(query_points, self.pc_range)
        geometry_mask = self.compute_attn_mask(query_points, gt_bboxes_3d, gt_labels_3d, dist)
        tau = tau.permute(0, 2, 1)
        attn_mask = geometry_mask[:, None, :, :] * tau[..., None]
        attn_mask = attn_mask.flatten(0, 1)
        return self.attention(query_feat, attn_mask=attn_mask)
    
    def forward(self, query_points, query_feat, gt_bboxes_3d, gt_labels_3d):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat, gt_bboxes_3d, gt_labels_3d,
                      use_reentrant=False)
        else:
            return self.inner_forward(query_points, query_feat, gt_bboxes_3d, gt_labels_3d)

    @torch.no_grad()
    def calc_points_dists(self, points):
        points = decode_points(points, self.pc_range)
        points = points.mean(dim=2)
        dist = torch.norm(points.unsqueeze(-2) - points.unsqueeze(-3), dim=-1)
        return -dist


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
        B, Q, G, T, P, C = x.shape
        assert G == self.n_groups
        assert T * P == self.in_points
        assert C == self.eff_in_dim
        
        '''generate temporal consistency parameters'''
        x_reshaped = x.reshape(B*Q, T, G*P, C)  # [B*Q, T, P*G, C]

        # Compute mean feature per frame for each query-group combination
        frame_features = x_reshaped.mean(dim=2)  # [B*Q, T, C]

        # Compute pairwise cosine similarity between frames
        frame_features_norm = F.normalize(frame_features, p=2, dim=-1)  # [B*Q, T, C]
        similarity_matrix = torch.matmul(frame_features_norm, frame_features_norm.transpose(1, 2))  # [B*Q, T, T]
        temporal_weights = F.softmax(similarity_matrix, dim=-1)[..., -1].reshape(B*Q, T, 1, 1)  # [B*Q, T, 1, 1]

        # Aggregate features across frames based on semantic similarity
        x_reshaped_weighted = temporal_weights.repeat(1, 1, G*P, 1) * x_reshaped  # [B*Q, T, G*P, C]
        x = x_reshaped_weighted.reshape(B, Q, G, T*P, C)

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        out = x.reshape(B*Q, G, T*P, C)

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

class PositionEncoderV0(nn.Module):
    """
    Position encoder enhanced with learned occupancy priors.

    Integrates:
    1. Fourier positional encoding for multi-scale spatial patterns
    2. Pre-computed occupancy probability distribution from training data
    3. Spatial entropy (uncertainty) information
    """
    def __init__(self,
                 embed_dims,
                 prior_path=None,
                 num_freq_bands=8,
                 grid_shape=(200, 200, 16),
                 pc_range=[],
                 use_trilinear=True):
        """
        Args:
            embed_dims: Embedding dimension
            prior_path: Path to pre-computed occupancy statistics (.pkl)
            num_freq_bands: Number of Fourier frequency bands
            grid_shape: Voxel grid dimensions (X, Y, Z)
            pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
            use_trilinear: Use trilinear interpolation for smooth lookup
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.grid_shape = grid_shape
        self.pc_range = torch.tensor(pc_range).reshape(2, 3) if len(pc_range) > 0 else None
        self.use_trilinear = use_trilinear
        self.num_freq_bands = num_freq_bands

        if prior_path is not None and os.path.exists(prior_path):
            with open(prior_path, 'rb') as f:
                stats = pickle.load(f)

            self.register_buffer(
                'occupancy_density', torch.from_numpy(stats['occupancy_density']).float())

            self.register_buffer(
                'entropy', torch.from_numpy(stats['entropy']).float())
        else:
            # Initialize with uniform priors if not available
            self.register_buffer(
                'occupancy_density', torch.ones(grid_shape) * 0.5
            )
            self.register_buffer(
                'entropy', torch.ones(grid_shape) * 0.0
            )
        # Learnable frequency bands for Fourier encoding
        self.freq_bands = nn.Parameter(
            torch.linspace(0, num_freq_bands - 1, num_freq_bands)
        )

        # Fourier feature projection
        self.fourier_proj = nn.Sequential(
            nn.Linear(3 * num_freq_bands * 2, embed_dims // 3),
            nn.LayerNorm(embed_dims // 3),
            nn.ReLU(inplace=True),
        )

        # Prior feature encoder (density + entropy)
        self.prior_encoder = nn.Sequential(
            nn.Linear(2, embed_dims // 3),
            nn.LayerNorm(embed_dims // 3),
            nn.ReLU(inplace=True),
        )

        # Absolute position encoder
        self.abs_pos_encoder = nn.Sequential(
            nn.Linear(3, embed_dims // 3),
            nn.LayerNorm(embed_dims // 3),
            nn.ReLU(inplace=True),
        )

        # Fusion module
        self.fusion = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

    def sample_prior_features(self, positions):
        """
        Sample occupancy prior features at given positions.

        Args:
            positions: (N, 3) normalized positions in [0, 1]

        Returns:
            density: (N,) occupancy density at each position
            entropy: (N,) spatial entropy at each position
        """
        N = positions.shape[0]
        device = positions.device

        # Convert normalized positions to grid indices
        # positions are in range [0, 1] for each dimension
        grid_indices = positions * torch.tensor(
            self.grid_shape, device=device, dtype=positions.dtype
        ).view(1, 3)

        if self.use_trilinear:
            # Trilinear interpolation for smooth lookup
            density = self._trilinear_sample(
                self.occupancy_density, grid_indices
            )
            entropy = self._trilinear_sample(
                self.entropy, grid_indices
            )
        else:
            # Nearest neighbor lookup
            grid_indices = torch.clamp(
                grid_indices.long(),
                min=torch.zeros(3, device=device).long(),
                max=torch.tensor(self.grid_shape, device=device).long() - 1
            )

            density = self.occupancy_density[
                grid_indices[:, 0],
                grid_indices[:, 1],
                grid_indices[:, 2]
            ]

            entropy = self.entropy[
                grid_indices[:, 0],
                grid_indices[:, 1],
                grid_indices[:, 2]
            ]

        return density, entropy

    def _trilinear_sample(self, grid, positions):
        """
        Trilinear interpolation for 3D grid sampling.

        Args:
            grid: (X, Y, Z) 3D grid
            positions: (N, 3) continuous positions in grid coordinates

        Returns:
            values: (N,) interpolated values
        """
        device = positions.device

        # Get integer and fractional parts
        positions_floor = torch.floor(positions).long()
        positions_frac = positions - positions_floor.float()

        # Clamp to valid range
        X, Y, Z = self.grid_shape
        positions_floor = torch.clamp(
            positions_floor,
            min=torch.zeros(3, device=device).long(),
            max=torch.tensor([X-2, Y-2, Z-2], device=device).long()
        )

        # Get 8 corner points
        x0, y0, z0 = positions_floor[:, 0], positions_floor[:, 1], positions_floor[:, 2]
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

        # Get fractional distances
        xd, yd, zd = positions_frac[:, 0], positions_frac[:, 1], positions_frac[:, 2]

        # Sample 8 corners
        c000 = grid[x0, y0, z0]
        c001 = grid[x0, y0, z1]
        c010 = grid[x0, y1, z0]
        c011 = grid[x0, y1, z1]
        c100 = grid[x1, y0, z0]
        c101 = grid[x1, y0, z1]
        c110 = grid[x1, y1, z0]
        c111 = grid[x1, y1, z1]

        # Trilinear interpolation
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        c = c0 * (1 - zd) + c1 * zd

        return c

    def forward(self, query_points):
        """
        Args:
            query_points: (B, Q, K, 3) normalized positions in [0, 1]

        Returns:
            pos_encoding: (B, Q, K, embed_dims) position encodings
        """
        # 1. Fourier encoding
        freq = self.freq_bands.view(1, 1, -1)
        pos = query_points.unsqueeze(-1)
        angle = 2 * torch.pi * pos * torch.exp2(freq)
        fourier_feats = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        fourier_feats = fourier_feats.flatten(-2)
        fourier_encoded = self.fourier_proj(fourier_feats)

        # 2. Sample occupancy priors
        density, entropy = self.sample_prior_features(positions)
        prior_feats = torch.stack([density, entropy], dim=-1)  # (N, 2)
        prior_encoded = self.prior_encoder(prior_feats)

        # 3. Absolute position encoding
        abs_pos_encoded = self.abs_pos_encoder(positions)

        # 4. Fuse all features
        combined = torch.cat([
            fourier_encoded,
            prior_encoded,
            abs_pos_encoded
        ], dim=-1)

        return self.fusion(combined)
