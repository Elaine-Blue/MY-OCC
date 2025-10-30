import numpy as np
import os.path as osp
from mmdet3d.datasets.builder import PIPELINES
from scipy import ndimage

@PIPELINES.register_module()
class BoundaryGTFilter:
    '''
    Generate boundary ground truth labels for occupancy prediction.

    Args:
        include_classes (list): Classes to compute boundaries for.
                               If None, compute for all non-empty classes.
        empty_label (int): Label for empty/free space. Default: 17
    '''
    def __init__(self,
                 include_classes=None,
                 threshold=0,
                 empty_label=17):
        self.include_classes = include_classes
        self.empty_label = empty_label
        self.threshold = threshold

    def compute_boundary_mask(self, voxel_semantics, mask_camera):
        """
        Simplified boundary detection using convolution.
        """ 
        label_mask = (voxel_semantics == self.empty_label).astype(np.uint8)
        
        kernel = np.zeros((3, 3, 3), dtype=np.uint8)
        kernel[1, 1, :] = 1  # front-back
        kernel[1, :, 1] = 1  # left-right  
        kernel[:, 1, 1] = 1  # up-down

        neighbor_count = ndimage.convolve(label_mask, kernel, mode='constant')
        boundary_mask = (neighbor_count > self.threshold).astype(np.uint8)
        boundary_mask &= mask_camera
        boundary_mask &= (voxel_semantics != self.empty_label)
        return boundary_mask

    def compute_boundary_labels(self, semantics, mask_camera=None):
        """
        Generate boundary and interior labels from semantic labels.

        Args:
            semantics: [H, W, D] array with semantic class labels
        Returns:
            boundary_labels: [H, W, D] array (1=boundary, 0=non-boundary)
        """
        boundary_labels = self.compute_boundary_mask(semantics, mask_camera)
        
        # Filter by include_classes if specified
        if self.include_classes is not None:
            class_mask = np.isin(semantics, self.include_classes)
            boundary_labels &= class_mask
        
        return boundary_labels

    def __call__(self, results):
        """
        Generate boundary and interior labels.

        Args:
            results: Dictionary containing:
                - 'voxel_semantics': [H, W, D] semantic labels

        Returns:
            results: Dictionary with added keys:
                - 'voxel_boundary': [H, W, D] boundary labels (0 or 1)
                - 'voxel_interior': [H, W, D] interior labels (0 or 1)
        """
        semantics = results['voxel_semantics']
        mask_camera = results['mask_camera']
        
        boundary_labels = self.compute_boundary_labels(semantics, mask_camera)

        results['boundary_labels'] = boundary_labels
        return results
