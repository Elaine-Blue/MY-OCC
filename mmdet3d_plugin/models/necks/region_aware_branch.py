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
class RegionAwareHead(BaseModule):
    '''
    Region-Aware Branch to predict vegetation, manmade, driveable_surface, terrian, sidewalk.
    '''
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 pc_range=[],
                 empty_label=17,
                 voxel_size=[],
                 target_classes=[],
                 use_priori_points=False,
                 clusters_data_path=None,
                 transformer=None,
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
        
        # TODO: EXP 1
        self.use_priori_points = use_priori_points
        self.clusters_data_path = clusters_data_path
        
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)

        self._init_layers()
        
    def _init_layers(self):
        if self.use_priori_points:
            with open(self.clusters_data_path, 'rb') as f:
                clusters_data = pickle.load(f)
            cluster_points = torch.cat([torch.from_numpy(v) for k, v in clusters_data.items()], dim=0)
            sample_idxes = torch.randint(0, cluster_points.shape[0], (self.num_query,))
            init_points = cluster_points[sample_idxes, :3].float()
            x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
            init_points[..., 0] = (init_points[..., 0] - x_min) / (x_max - x_min)
            init_points[..., 1] = (init_points[..., 1] - y_min) / (y_max - y_min)
            init_points[..., 2] = (init_points[..., 2] - z_min) / (z_max - z_min)
            self.init_points = nn.Embedding.from_pretrained(init_points, freeze=False)
        else:
            self.init_points = nn.Embedding(self.num_query, 3)
            nn.init.uniform_(self.init_points.weight, 0, 1)
        init_points_dim = self.init_points.weight.shape[1]
        self.query_feat = nn.Linear(init_points_dim, self.embed_dims)
        
    def init_weights(self):
        self.transformer.init_weights()

    def forward(self, mlvl_feats, img_metas):
        B, Q = mlvl_feats[0].shape[0], self.num_query
        query_feat = self.query_feat(self.init_points.weight[None, :, :].repeat(B, 1, 1))
        init_points = self.init_points.weight[None, :, None, :3].repeat(B, 1, 1, 1)
    
        cls_scores, refine_pts = self.transformer(
            init_points,
            query_feat,
            mlvl_feats,
            img_metas=img_metas,
        )
        return init_points, cls_scores, refine_pts
