import numpy as np
import os.path as osp
from mmdet3d.datasets.builder import PIPELINES

@PIPELINES.register_module()
class BoundaryGTFilter:
    '''
    Generate boundary ground truth labels for occupancy prediction.

    For each voxel, generates a 6-directional boundary encoding:
    - State 0: Same semantic class (interior)
    - State 1: Occupied-to-free boundary
    - State 2: Occupied-to-occupied boundary (different classes)

    Args:
        include_classes (list): Classes to compute boundaries for.
                               If None, compute for all non-empty classes.
        empty_label (int): Label for empty/free space. Default: 17
        mode (str): 'binary' (old) or 'directional' (new 6-dim encoding)
    '''
    def __init__(self,
                 include_classes=None,
                 empty_label=17,
                 mode='directional'):
        self.include_classes = include_classes
        self.empty_label = empty_label
        self.mode = mode

    def compute_directional_boundary(self, voxel_semantics, mask_camera):
        """
        Compute 6-directional boundary states for each voxel.

        Returns:
            boundary_6d: [H, W, D, 6] array with values:
                - 0: Same semantic class
                - 1: Occupied-to-free boundary
                - 2: Occupied-to-occupied with different class

            Order: [+x, -x, +y, -y, +z, -z] (right, left, back, front, down, up)
        """
        H, W, D = voxel_semantics.shape
        boundary_6d = np.zeros((H, W, D, 6), dtype=np.uint8)

        # Define 6 directions: [dx, dy, dz]
        # Order: +x, -x, +y, -y, +z, -z
        directions = [
            (1, 0, 0),   # right (+x)
            (-1, 0, 0),  # left (-x)
            (0, 1, 0),   # back (+y)
            (0, -1, 0),  # front (-y)
            (0, 0, 1),   # up (+z)
            (0, 0, -1)   # down (-z)
        ]

        for dir_idx, (dx, dy, dz) in enumerate(directions):
            # Compute shifted semantics for this direction
            if dx == 1:
                center_slice = (slice(None, -1), slice(None), slice(None))
                neighbor_slice = (slice(1, None), slice(None), slice(None))
            elif dx == -1:
                center_slice = (slice(1, None), slice(None), slice(None))
                neighbor_slice = (slice(None, -1), slice(None), slice(None))
            elif dy == 1:
                center_slice = (slice(None), slice(None, -1), slice(None))
                neighbor_slice = (slice(None), slice(1, None), slice(None))
            elif dy == -1:
                center_slice = (slice(None), slice(1, None), slice(None))
                neighbor_slice = (slice(None), slice(None, -1), slice(None))
            elif dz == 1:
                center_slice = (slice(None), slice(None), slice(None, -1))
                neighbor_slice = (slice(None), slice(None), slice(1, None))
            else:  # dz == -1
                center_slice = (slice(None), slice(None), slice(1, None))
                neighbor_slice = (slice(None), slice(None), slice(None, -1))
            
            center = voxel_semantics[center_slice]
            neighbor = voxel_semantics[neighbor_slice]

            # State 0: Same class (default, already initialized to 0)
            same_class = (center == neighbor)

            # State 1: Center occupied, neighbor free
            center_occupied = (center != self.empty_label)
            neighbor_free = (neighbor == self.empty_label)
            occ_to_free = center_occupied & neighbor_free

            # State 2: Both occupied, different class
            neighbor_occupied = (neighbor != self.empty_label)
            both_occ_diff = center_occupied & neighbor_occupied & (~same_class)

            # Assign states
            boundary_6d[center_slice + (dir_idx,)][occ_to_free] = 1
            boundary_6d[center_slice + (dir_idx,)][both_occ_diff] = 2

        # Apply camera mask filter (only keep visible voxels)
        mask_expanded = mask_camera[..., None].repeat(6, axis=-1)  # [H, W, D, 6]
        boundary_6d = boundary_6d * mask_expanded

        return boundary_6d

    def compute_boundary_mask(self, voxel_semantics, mask_camera):
        """
        Legacy binary boundary detection (backward compatibility).
        A voxel is a boundary if any of its 6-connected neighbors has a different label.
        """
        boundary_mask = np.zeros_like(voxel_semantics, dtype=bool)
        
        # X-axis (left-right)
        boundary_mask[1:, :, :] |= (voxel_semantics[1:, :, :] != voxel_semantics[:-1, :, :])
        boundary_mask[:-1, :, :] |= (voxel_semantics[:-1, :, :] != voxel_semantics[1:, :, :])

        # Y-axis (front-back)
        boundary_mask[:, 1:, :] |= (voxel_semantics[:, 1:, :] != voxel_semantics[:, :-1, :])
        boundary_mask[:, :-1, :] |= (voxel_semantics[:, :-1, :] != voxel_semantics[:, 1:, :])

        # Z-axis (up-down)
        boundary_mask[:, :, 1:] |= (voxel_semantics[:, :, 1:] != voxel_semantics[:, :, :-1])
        boundary_mask[:, :, :-1] |= (voxel_semantics[:, :, :-1] != voxel_semantics[:, :, 1:])

        # Apply camera mask filter
        boundary_mask = boundary_mask.astype(np.uint8) & mask_camera
        boundary_mask = boundary_mask & (voxel_semantics != self.empty_label)
        return boundary_mask

    def compute_boundary_free_voxels(self, voxel_semantics, mask_camera):
        """
        Extract free voxels that are adjacent to occupied voxels (boundary free).

        These are hard negative samples: free space near objects.

        Returns:
            boundary_free_mask: [H, W, D] bool array
                True for free voxels adjacent to occupied voxels
        """

        # Identify VISIBLE free and occupied voxels only
        # This ensures we only check neighbors that are inside the camera frustum
        visible_free_mask = voxel_semantics == self.empty_label
        visible_occupied_mask = voxel_semantics != self.empty_label

        # Find free voxels that have at least one VISIBLE occupied neighbor
        boundary_free_mask = np.zeros_like(voxel_semantics, dtype=bool)

        # Check each of 6 directions
        # X-axis
        boundary_free_mask[1:, :, :] |= visible_free_mask[1:, :, :] & visible_occupied_mask[:-1, :, :]
        boundary_free_mask[:-1, :, :] |= visible_free_mask[:-1, :, :] & visible_occupied_mask[1:, :, :]

        # Y-axis
        boundary_free_mask[:, 1:, :] |= visible_free_mask[:, 1:, :] & visible_occupied_mask[:, :-1, :]
        boundary_free_mask[:, :-1, :] |= visible_free_mask[:, :-1, :] & visible_occupied_mask[:, 1:, :]

        # Z-axis
        boundary_free_mask[:, :, 1:] |= visible_free_mask[:, :, 1:] & visible_occupied_mask[:, :, :-1]
        boundary_free_mask[:, :, :-1] |= visible_free_mask[:, :, :-1] & visible_occupied_mask[:, :, 1:]

        boundary_free_mask &= mask_camera.astype(bool)
        return boundary_free_mask

    def compute_boundary_labels(self, semantics, mask_camera):
        """
        Generate boundary labels from semantic labels.

        Returns:
            - If mode='directional': [H, W, D, 6] array with 3-state encoding
            - If mode='binary': [H, W, D] binary array
        """
        if self.mode == 'directional':
            boundary_labels = self.compute_directional_boundary(semantics, mask_camera)
        else:
            boundary_labels = self.compute_boundary_mask(semantics, mask_camera)

        # Filter by include_classes if specified
        if self.include_classes is not None:
            class_mask = np.isin(semantics, self.include_classes)
            if self.mode == 'directional':
                class_mask_expanded = class_mask[..., None].repeat(6, axis=-1)
                boundary_labels = boundary_labels * class_mask_expanded
            else:
                boundary_labels = boundary_labels & class_mask

        return boundary_labels

    def __call__(self, results):
        """
        Generate boundary and interior labels.

        Returns:
            results: Dictionary with added keys:
                - 'boundary_labels': [H, W, D, 6] directional boundary labels (0/1/2)
                - 'voxel_semantics_filter': [H, W, D] semantics with filtered free space
                    - boundary free voxels: 17
                    - interior free voxels: 18
        """
        semantics = results['voxel_semantics']
        mask_camera = results['mask_camera']
        results['voxel_semantics_raw'] = semantics.copy()
        
        boundary_labels = self.compute_boundary_labels(semantics, mask_camera)

        # Create filtered semantics for training (17 classes, ignoring interior free)
        voxel_semantics_filter = semantics.copy()

        # Get boundary free voxels (free voxels adjacent to occupied voxels)
        boundary_free_mask = self.compute_boundary_free_voxels(semantics, mask_camera)
        
        # Identify all free voxels in original semantics
        all_free_mask = (semantics == self.empty_label).astype(bool)

        # Interior free voxels = all free voxels - boundary free voxels
        interior_free_mask = all_free_mask & (~boundary_free_mask.astype(bool))
        voxel_semantics_filter[interior_free_mask] = 18
        
        results['boundary_labels'] = boundary_labels
        results['voxel_semantics'] = voxel_semantics_filter
        return results

if __name__ == '__main__':
    import sys
    sys.path.append('/root/autodl-tmp/Project/MY-OCC')

    # Load GT data
    gt_path = 'data/nuscenes/gts/scene-0123/7c36879f3ba04809976c39fd03a8a01c/labels.npz'
    data = np.load(gt_path)

    # Prepare input for BoundaryGTFilter
    results = {
        'voxel_semantics': data['semantics'],
        'mask_camera': data['mask_camera']
    }

    boundary_filter_6d = BoundaryGTFilter(
        empty_label=17,
        # include_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ],
        mode='directional'
    )

    results_6d = boundary_filter_6d(results.copy())
    boundary_6d = results_6d['boundary_labels']

    print("Directional Boundary Statistics:")
    print("-"*70)

    direction_names = ['Right(+X)', 'Left(-X)', 'Back(+Y)', 'Front(-Y)', 'Up(+Z)', 'Down(-Z)']
    state_names = ['Same Class', 'Occ->Free', 'Occ->Occ']

    for dir_idx, dir_name in enumerate(direction_names):
        print(f"\n{dir_name}:")
        for state in range(3):
            count = (boundary_6d[..., dir_idx] == state).sum()
            print(f"  State {state} ({state_names[state]:12s}): {count:7d} voxels")

    from tools.visualizer import Visualizer
    import cv2

    voxel_size = 0.4
    visualizer = Visualizer()
    # Visualize the voxels outside the camera frustum
    semantics = results['voxel_semantics_raw']
    mask_camera = results['mask_camera'].astype(bool)

    semantics_viz = results_6d['voxel_semantics']
    boundary_mask = ((semantics_viz != 18) & data['mask_camera']).astype(bool)
    print("Total predicted voxels number:", boundary_mask.sum())
    img = visualizer.draw_gt_voxels_3D(
        semantics_viz,
        boundary_mask,
        voxel_size=voxel_size,
        filter_classes=None
    )
    cv2.imwrite('boundary_directional.png', img)
    print("✓ Saved visualization: boundary_directional.png")