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
