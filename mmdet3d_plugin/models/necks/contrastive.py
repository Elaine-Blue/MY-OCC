import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from mmcv.ops import knn
from ..bbox.utils import decode_points


class RegionConsistencyLoss(nn.Module):
    """
        Region Consistency Module for enhancing feature representation.
        Contrast different regions based on spatial and semantic consistency.
    """
    
    def __init__(self, 
                 embed_dim=256, 
                 temperature=0.1, 
                 pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], 
                 num_classes=17,
                 min_cosine_similarity=0.5,
                 region_stride=100,
                 region_voxel_size = [2.0, 2.0, 0.8]
                ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.pc_range = pc_range
        self.num_classes = num_classes
        self.min_cosine_similarity = min_cosine_similarity
        self.region_voxel_size = region_voxel_size
        self.region_stride = region_stride
        self.region_size = [
            (self.pc_range[3] - self.pc_range[0]) / self.region_voxel_size[0],
            (self.pc_range[4] - self.pc_range[1]) / self.region_voxel_size[1],
            (self.pc_range[5] - self.pc_range[2]) / self.region_voxel_size[2]
        ]

    def partition_into_regions(self, query_points):
        """
        Split query points into spatial regions.
        Args:
            query_points: [B, N, 3] normalized world coordinates
        Returns:
            region_ids: [B, N] Region identifiers for each point
        """
        voxel_coords = torch.zeros_like(query_points)
        voxel_coords[..., 0] = torch.floor((query_points[..., 0] - self.pc_range[0]) / self.region_voxel_size[0]).long()
        voxel_coords[..., 1] = torch.floor((query_points[..., 1] - self.pc_range[1]) / self.region_voxel_size[1]).long()
        voxel_coords[..., 2] = torch.floor((query_points[..., 2] - self.pc_range[2]) / self.region_voxel_size[2]).long()
        voxel_coords[..., 0] = torch.clamp(voxel_coords[..., 0], 0.0, self.region_size[0])
        voxel_coords[..., 1] = torch.clamp(voxel_coords[..., 1], 0.0, self.region_size[1])
        voxel_coords[..., 2] = torch.clamp(voxel_coords[..., 2], 0.0, self.region_size[2])
        
        region_stride = self.region_stride
        region_ids = voxel_coords * torch.tensor(
            [region_stride * region_stride, region_stride, 1], 
            device=query_points.device
        )
        region_ids = region_ids.sum(dim=-1)
        return region_ids

    def extract_region_valid_labels(self, region_id, voxel_semantics, mask_camera):
        x, y, z = self.region_id_to_coord(region_id)
        X, Y, Z = voxel_semantics.shape
        rescale = [int(X // self.region_size[0]), int(Y // self.region_size[1]), int(Z // self.region_size[2])]
        region_semantics = voxel_semantics[x:x+rescale[0], y:y+rescale[1], z:z+rescale[2]]
        region_mask = mask_camera[x:x+rescale[0], y:y+rescale[1], z:z+rescale[2]]
        return region_semantics, region_mask
        
    def region_id_to_coord(self, region_id):
        region_stride = self.region_stride
        z = region_id % region_stride
        remaining = region_id // region_stride
        y = remaining % region_stride
        x = remaining // region_stride
        return int(x), int(y), int(z)
    
    def aggregate_region_features(self, 
                                  query_feats, 
                                  query_points, 
                                  voxel_semantics, 
                                  mask_camera, 
                                  region_ids):
        """
        Args:
            query_feats: [B, N, C]
            query_points: [B, N, 3]
            voxel_semantics: [B, 200, 200, 16]
            mask_camera: [B, 200, 200, 16]
            region_ids: [B, N]
        
        Returns:
            region_features: [B, R, C]
            region_semantics: [B, R]
            region_centers: [B, R, 3]
        """
        B, Q, embed_dim = query_feats.shape
        device = query_feats.device
        all_region_features = []
        all_region_semantics = []
        all_region_centers = []
        
        for i in range(B):
            regions_ids_i = region_ids[i]
            query_feats_i = query_feats[i]
            query_points_i = query_points[i]
            voxel_labels_i = voxel_semantics[i]
            voxel_labels_mask_i = mask_camera[i]
            
            unique_regions, region_indices = torch.unique(regions_ids_i, return_inverse=True)
            
            batch_region_features = []
            batch_region_semantics = []
            batch_region_centers = []
            
            for i, region_id in enumerate(unique_regions):
                region_mask = (regions_ids_i == region_id)
                region_points = query_points_i[region_mask]
                region_feats = query_feats_i[region_mask]
                region_sems, region_mask = self.extract_region_valid_labels(region_id, voxel_labels_i, voxel_labels_mask_i)
                
                if len(region_points) == 0:
                    continue
                
                region_center = region_points.mean(dim=0)
                dominant_semantic = torch.mode(region_sems[region_mask]).values
                aggregated_feature = (region_feats).sum(dim=0)
                batch_region_features.append(aggregated_feature)
                batch_region_semantics.append(dominant_semantic.unsqueeze(0))
                batch_region_centers.append(region_center)
            
            if batch_region_features:
                batch_region_features = torch.stack(batch_region_features, dim=0)
                batch_region_semantics = torch.stack(batch_region_semantics, dim=0)
                batch_region_centers = torch.stack(batch_region_centers, dim=0)
            else:
                batch_region_features = torch.zeros((0, embed_dim), device=device)
                batch_region_semantics = torch.zeros((0,), device=device, dtype=torch.long)
                batch_region_centers = torch.zeros((0, 3), device=device)
            
            all_region_features.append(batch_region_features)
            all_region_semantics.append(batch_region_semantics)
            all_region_centers.append(batch_region_centers)
        
        return all_region_features, all_region_semantics, all_region_centers

    def find_consistent_region_pairs(self, 
                                     region_centers,
                                     region_semantics,
                                     num_pos_pairs=200):
        """
        Find consistent region pairs for contrastive learning.
        Args:
            region_centers: List[[R_i, 3]]
            region_semantics: List[[R_i]]
        Returns:
            pos_pairs: List[List[(i, j)]]
        """
        batch_size = len(region_centers)
        all_pos_pairs = []
        
        for b in range(batch_size):
            centers = region_centers[b]  # [R, 3]
            semantics = region_semantics[b]  # [R]
            
            num_regions = len(centers)
            if num_regions < 2:
                all_pos_pairs.append([])
                continue
            
            labels = F.one_hot(semantics.long(), num_classes=self.num_classes).squeeze(1)
            concat_features = torch.cat([centers, labels], dim=1)
            candidate_pairs = []
            
            for i in range(num_regions):
                j = knn(2, concat_features.unsqueeze(0), concat_features[i:i+1].unsqueeze(0))[0, 1, 0]
                score = F.cosine_similarity(concat_features[i].unsqueeze(0), concat_features[j].unsqueeze(0), dim=-1)
                
                if score > self.min_cosine_similarity:
                    candidate_pairs.append((i, j, score.item()))
            
            if len(candidate_pairs) > 0:
                candidate_pairs.sort(key=lambda x: x[2], reverse=True)
                selected_pairs = candidate_pairs[:num_pos_pairs]
                all_pos_pairs.append(selected_pairs)
            else:
                all_pos_pairs.append([])
        
        return all_pos_pairs

    def loss(self, region_features, pos_pairs):
        """
        Compute contrastive loss for region features.
        Args:
            region_features: List[[R_i, C]]
            pos_pairs: List[List[(i, j)]]
        Returns:
            loss: torch.Tensor
        """
        batch_size = len(region_features)
        total_loss = 0.0
        total_pairs = 0
        
        for b in range(batch_size):
            features = region_features[b]  # [R, C]
            batch_pairs = pos_pairs[b]
            
            if len(features) < 2 or len(batch_pairs) == 0:
                continue

            features = F.normalize(features, p=2, dim=-1)
            
            similarity_matrix = torch.matmul(features, features.T)  # [R, R]
            
            batch_loss = 0.0
            for anchor_idx, positive_idx, _ in batch_pairs:
                pos_sim = similarity_matrix[anchor_idx, positive_idx] / self.temperature
                
                neg_mask = torch.ones(len(features), dtype=torch.bool, device=features.device)
                neg_mask[anchor_idx] = False
                neg_mask[positive_idx] = False
                
                if neg_mask.sum() == 0:
                    continue
                
                neg_sims = similarity_matrix[anchor_idx, neg_mask] / self.temperature
                
                # InfoNCE Loss
                numerator = torch.exp(pos_sim)
                denominator = numerator + torch.exp(neg_sims).sum()
                
                if denominator > 0:
                    pair_loss = -torch.log(numerator / denominator)
                    batch_loss += pair_loss
                    total_pairs += 1
            
            if len(batch_pairs) > 0:
                total_loss += batch_loss / len(batch_pairs)
        
        return total_loss / batch_size if total_pairs > 0 else torch.tensor(0.0, device=region_features[0].device)

    def forward(self, query_feats, query_points, voxel_semantics, mask_camera, num_pos_pairs=200):
        """
        Args:
            query_feats: [B, N, C] 
            query_points: [B, N, 3]
            voxel_semantics: [B, 200, 200, 16]
            mask_camera: [B, 200, 200, 16]
        Returns:
            contrastive_loss
        """
        region_ids = self.partition_into_regions(decode_points(query_points, self.pc_range))
        
        region_features, region_semantics, region_centers = self.aggregate_region_features(
            query_feats, query_points, voxel_semantics, mask_camera, region_ids
        )

        pos_pairs = self.find_consistent_region_pairs(region_centers, region_semantics, num_pos_pairs)
        contrastive_loss = self.loss(region_features, pos_pairs)
        return contrastive_loss



if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda:0'
    batch_size = 2
    num_points = 2000
    embed_dim = 256
    
    query_feats = torch.randn(batch_size, num_points, embed_dim).to(device)
    voxel_semantics = torch.randint(0, 5, (batch_size, 200, 200, 16)).float().to(device)
    mask_camera = torch.randint(0, 2, (batch_size, 200, 200, 16)).bool().to(device)
    
    query_points = torch.rand(batch_size, num_points, 3).to(device)
    
    contrastive_module = RegionConsistencyLoss(
        embed_dim,
        temperature=0.1,
        num_classes=16,
        min_cosine_similarity=0.01
    )

    loss = contrastive_module(query_feats, query_points, voxel_semantics, mask_camera)
    print('contrastive loss : ', loss)