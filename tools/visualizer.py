import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tools.mayaviOffScreen import mlab
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import torch

classname_to_color = {  # RGB.
    0: (0, 0, 0),  # Black. noise
    1: (112, 128, 144),  # Slategrey barrier
    2: (220, 20, 60),  # Crimson bicycle
    3: (255, 127, 80),  # Orangered bus
    4: (255, 158, 0),  # Orange car
    5: (233, 150, 70),  # Darksalmon construction
    6: (255, 61, 99),  # Red motorcycle
    7: (0, 0, 230),  # Blue pedestrian
    8: (47, 79, 79),  # Darkslategrey trafficcone
    9: (255, 140, 0),  # Darkorange trailer
    10: (255, 99, 71),  # Tomato truck
    11: (0, 207, 191),  # nuTonomy green driveable_surface
    12: (175, 0, 75),  # flat other
    13: (75, 0, 75),  # sidewalk
    14: (112, 180, 60),  # terrain
    15: (222, 184, 135),  # Burlywood mannade
    16: (0, 175, 0),  # Green vegetation
    17: (230, 230, 230),  # boundary free
    18: (230, 230, 230), # interior free 
}

occ_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free', 'boundary'
]

color_map = {
    key: (color[2]/255.0, color[1]/255.0, color[0]/255.0)
    for key, color in classname_to_color.items()
}

def world_to_voxel(
        points, 
        voxel_size=[0.4, 0.4, 0.4], 
        pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    ):
    points = points.copy()
    points[..., 0] = np.clip(points[..., 0], pc_range[0], pc_range[3] - voxel_size[0])
    points[..., 1] = np.clip(points[..., 1], pc_range[1], pc_range[4] - voxel_size[1])
    points[..., 2] = np.clip(points[..., 2], pc_range[2], pc_range[5] - voxel_size[2])
    
    points[..., 0] = (points[..., 0] - pc_range[0]) / voxel_size[0]
    points[..., 1] = (points[..., 1] - pc_range[1]) / voxel_size[1]
    points[..., 2] = (points[..., 2] - pc_range[2]) / voxel_size[2]
    
    return points.astype(np.int32)

def get_sparse_voxels(
        voxel_semantics, 
        mask_camera, 
        pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        scene_size=[80, 80, 6.4],
        filter_classes=[17]
    ):
    '''
        Args:
            voxel_semantics: (W, H, Z) tensor / numpy array
            mask_camera: (W, H, Z) tensor / numpy array
            filter_classes: list of class indices to filter
    '''
    if isinstance(voxel_semantics, np.ndarray):
        voxel_semantics = torch.from_numpy(voxel_semantics)
        
    W, H, Z = voxel_semantics.shape
    voxel_semantics = voxel_semantics.long()
    
    x = torch.arange(0, W, dtype=torch.float32)
    x = (x + 0.5) / W * scene_size[0] + pc_range[0]
    y = torch.arange(0, H, dtype=torch.float32)
    y = (y + 0.5) / H * scene_size[1] + pc_range[1]
    z = torch.arange(0, Z, dtype=torch.float32)
    z = (z + 0.5) / Z * scene_size[2] + pc_range[2]

    xx = x[:, None, None].expand(W, H, Z)
    yy = y[None, :, None].expand(W, H, Z)
    zz = z[None, None, :].expand(W, H, Z)
    world_coords = torch.stack([xx, yy, zz], dim=-1) # actual space
    if filter_classes is not None:
        mask = ~torch.isin(voxel_semantics, torch.tensor(filter_classes))
        mask = mask & mask_camera
    
    filter_coords = world_coords[mask].cpu().numpy()
    filter_labels = voxel_semantics[mask].cpu().numpy()
    return filter_coords, filter_labels

class Visualizer:
    """3D voxel Matched Results Visualizer"""
    def __init__(self, 
                 figsize=(16, 6), 
                 pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                 voxel_size=[0.4, 0.4, 0.4]
        ):
        self.figsize = figsize
        self.color_map = color_map
        self.class_names = occ_names
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.RESIZE_SAHPE = (600, 340)
        self.H = self.RESIZE_SAHPE[0] * 1.5
        self.W = self.RESIZE_SAHPE[0] * 1.5
        self.counter = 0
        self.empty_label = 17  # free space
        self.palette = np.array([classname_to_color[i] for i in range(len(classname_to_color))])

    def create_color_legend(self, SL=20, LW=20, TL=180, font_size=20, mode='row'):
        '''
            Args:
                SL: square length ;
                LW: text width
                TL: text length
        '''
        color_mapping = {name: color for name, color in zip(occ_names, classname_to_color.values())}
        cmap = color_mapping.copy()
        cmap.pop('others')
        if mode == 'row':
            row, col = 2, len(cmap) // 2
        else:
            row, col = len(color_mapping) // 2, 2
        w, h = int((SL+TL)*col), int(LW*(row*1.5-0.5))
        legend = Image.new('RGB', (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(legend)
        for i, (name, color) in enumerate(color_mapping.items()):
            row_idx, col_idx = i // col, i % col
            start_x_cube, start_y_cube = SL*col_idx*1.5+TL*col_idx, LW*row_idx*1.5
            if os.path.exists('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf'):
                font = ImageFont.truetype('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf', font_size)
            else:
                font = ImageFont.truetype('arial.ttf', font_size)
            draw.rectangle([start_x_cube, start_y_cube, start_x_cube+SL, start_y_cube+LW], fill=color)
            draw.text([start_x_cube+SL+5, start_y_cube], name, fill=(0, 0, 0), font=font)
        return np.array(legend)[:, :, ::-1]

    def downsample(self, points, factor):
        if factor <= 1:
            return points
        return points[::factor]
    
    def plot_matched_results_3D(self, ax, pred_pts, gt_pts, pred_labels, gt_labels, factor=8):
        """Draw Sparse Overview in 3D Space"""
        sparse_pred_pts = self.downsample(pred_pts, factor)
        sparse_gt_pts = self.downsample(gt_pts, factor)
        sparse_pred_labels = self.downsample(pred_labels, factor)
        sparse_gt_labels = self.downsample(gt_labels, factor)
        
        for idx, color in enumerate(self.color_map):
            pred_mask = (sparse_pred_labels == idx)
            gt_mask = (sparse_gt_labels == idx)
            class_name = self.class_names[idx]
            if pred_mask.any():
                ax.scatter(sparse_pred_pts[pred_mask, 0], sparse_pred_pts[pred_mask, 1], sparse_pred_pts[pred_mask, 2], 
                        c=color, alpha=0.7, s=10, label=f'{class_name} DTs', marker='o')
            if gt_mask.any():
                ax.scatter(sparse_gt_pts[gt_mask, 0], sparse_gt_pts[gt_mask, 1], sparse_gt_pts[gt_mask, 2], 
                    c=color, alpha=0.7, s=10, label=f'{class_name} GTs', marker='^')
          
        for pred_pt, gt_pt in zip(sparse_pred_pts, sparse_gt_pts):
            ax.plot([pred_pt[0], gt_pt[0]], 
                    [pred_pt[1], gt_pt[1]], 
                    [pred_pt[2], gt_pt[2]], 
                    color='lightgray', alpha=0.7, linewidth=1)
        
        ax.legend(bbox_to_anchor=(-0.5, 1.0), loc='upper left')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Sample Matched Pairs Overview ({factor}x Downsample)\n'
                    f'Pred: {len(sparse_pred_pts):,} points, GT: {len(sparse_gt_pts):,} points')


    def plot_matched_results_2D(self, ax, pred_pts, gt_pts, pred_labels, gt_labels, axis=[0, 1], factor=8):
        """Draw Sparse Overview in 2D Space."""
        sparse_pred_pts = self.downsample(pred_pts, factor)
        sparse_gt_pts = self.downsample(gt_pts, factor)
        sparse_pred_labels = self.downsample(pred_labels, factor)
        sparse_gt_labels = self.downsample(gt_labels, factor)
        
        for idx, color in enumerate(self.color_map):
            pred_mask = (sparse_pred_labels == idx)
            gt_mask = (sparse_gt_labels == idx)
            class_name = self.class_names[idx]
            if pred_mask.any():
                ax.scatter(sparse_pred_pts[pred_mask, axis[0]], sparse_pred_pts[pred_mask, axis[1]],
                        c=color, alpha=0.9, s=20, label=f'{class_name} DTs', marker='o')
            if gt_mask.any():
                ax.scatter(sparse_gt_pts[gt_mask, axis[0]], sparse_gt_pts[gt_mask, axis[1]],
                    c=color, alpha=0.9, s=20, label=f'{class_name} GTs', marker='^')
          
        for pred_pt, gt_pt in zip(sparse_pred_pts, sparse_gt_pts):
            ax.plot([pred_pt[0], gt_pt[0]], 
                    [pred_pt[1], gt_pt[1]], 
                    color='lightgray', alpha=0.9, linewidth=1)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        axis_names = ['X', 'Y', 'Z']
        ax.set_title(f'{axis_names[axis[0]]} - {axis_names[axis[1]]} 2D Projection ({factor}x Downsample)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def draw_matched_results(self, pred_pts, gt_pts, pred_labels, gt_labels, factor=8, target_classes=None, save_path=None):
        """
            Main function to visualize the matched results.
            Args:
                pred_pts (np.ndarray): [K, 3].
                gt_pts (np.ndarray): [K, 3].
                pred_labels (np.ndarray): [K, ].
                gt_labels (np.ndarray): [K, ].
                save_path (str, optional): The path to save the visualization image. Defaults to None.
                
            Layout:
                |---------------|---------------|---------------|
                |               |               |               |  
                |  3D Overview  |  2D Overview  |  2D Overview  | 
                |               |               |               |
                |---------------|---------------|---------------|
        """
        # Create Total Figure
        fig = plt.figure(figsize=self.figsize)
        
        # Update color_map & class name
        if target_classes is not None:
            self.color_map = [color_map[class_id] for class_id in target_classes]
            self.class_names = [occ_names[class_id] for class_id in target_classes]
        
        # 1. Draw 3D Overview
        ax1 = fig.add_subplot(121, projection='3d')
        self.plot_matched_results_3D(ax1, pred_pts, gt_pts, pred_labels, gt_labels, factor)
        
        # 2. Draw 2D Overview(z Axis)
        ax2 = fig.add_subplot(122)
        self.plot_matched_results_2D(ax2, pred_pts, gt_pts, pred_labels, gt_labels, [0, 1], factor)
        
        if save_path:
            plt.savefig(save_path, dpi=500, facecolor='white')
        else:
            fig.set_dpi(500)
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img_array = np.asarray(buf)
            return img_array
    
    
    def create_voxel_cube(self, center, size):
        """Create a voxel cube with given center and size."""
        x, y, z = center
        s = size / 2
        
        vertices = np.array([
            [x-s, y-s, z-s],  # 0
            [x+s, y-s, z-s],  # 1
            [x+s, y+s, z-s],  # 2
            [x-s, y+s, z-s],  # 3
            [x-s, y-s, z+s],  # 4
            [x+s, y-s, z+s],  # 5
            [x+s, y+s, z+s],  # 6
            [x-s, y+s, z+s],  # 7
        ])

        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 5, 6], [4, 6, 7],  # Top
            [0, 1, 5], [0, 5, 4],  # Front
            [2, 3, 7], [2, 7, 6],  # Back
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 2, 6], [1, 6, 5],  # Right
        ])
        
        return vertices, faces

    def draw_voxels_3D(self, x, y, z, labels, voxel_size, mode='cube'):
        """
        Draw Voxels in 3D space using Plotly.
        
        Parameters:
            - x, y, z: (n, ) world coordinates of voxels
            - labels: (n, ) voxel semantic labels
            - voxel_size: Float
        """
        classes = list(classname_to_color.keys())
        palette = self.palette
        if palette.shape[1] == 3:
            palette = np.concatenate([palette, np.ones((palette.shape[0], 1)) * 255], axis=1)
        fig = mlab.figure(size=(self.W, self.H), bgcolor=(1, 1, 1))

        plot = mlab.points3d(x, y, z,
                            labels,
                            scale_factor=voxel_size,
                        mode=mode,
                        scale_mode = "vector",
                        opacity=1.0,
                        vmin=1.0,
                        vmax=len(classes)-1)
        plot.module_manager.scalar_lut_manager.lut.table = palette
        
        # Disable interpolation to get exact palette colors
        plot.actor.property.interpolation = 'flat'
        plot.actor.mapper.interpolate_scalars_before_mapping = False

        f = mlab.gcf()
        f.scene._lift()

        save_fig = mlab.screenshot()
        mlab.close()
        return save_fig[:, :, ::-1]

    def draw_gt_voxels_3D(self, voxels, mask_camera, voxel_size=0.4, filter_classes=[17]):
        """Draw GT Voxels in 3D space using Plotly."""
        world_coords, labels = get_sparse_voxels(
            voxels, 
            mask_camera,
            filter_classes=filter_classes,
        )
        x, y, z = world_coords[:, 0], world_coords[:, 1], world_coords[:, 2]
        return self.draw_voxels_3D(x, y, z, labels, voxel_size)


    def vis_single(self, result, img_metas, save_dir, sample_token, add_legend=True):
        W = int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0])
        H = int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1])
        Z = int((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2])

        x = (np.arange(0, W) + 0.5) * self.voxel_size[0] + self.pc_range[0]
        y = (np.arange(0, H) + 0.5) * self.voxel_size[1] + self.pc_range[1]
        z = (np.arange(0, Z) + 0.5) * self.voxel_size[2] + self.pc_range[2]
        xx = x[:, None, None].repeat(H, axis=1).repeat(Z, axis=2)
        yy = y[None, :, None].repeat(W, axis=0).repeat(Z, axis=2)
        zz = z[None, None, :].repeat(W, axis=0).repeat(H, axis=1)
        
        scene_name = img_metas['scene_name']
        
        # 1. Visualize Camera Images.
        img_paths = img_metas['filename'][:6]
        cam_turn_map = [2, 0, 1, 4, 3, 5]
        img_paths_fix = [
            img_paths[idx] for idx in cam_turn_map
        ]
        resize_imgs = []
        camera_names = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        for camera_name, img_path in zip(camera_names, img_paths_fix):
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.RESIZE_SAHPE)
            cv2.putText(img, camera_name, (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            resize_imgs.append(img)
        
        # 2. Visualize GT Labels
        occ_labels = result['voxel_semantics']
        mask_camera = result['mask_camera']
        mask = occ_labels != self.empty_label
        x, y, z = xx[mask], yy[mask], zz[mask]
        label = occ_labels[mask].astype(np.int64)
        gt_sem_image = self.draw_voxels_3D(
            x, y, z,
            label,
            0.4,
        )
        
        # 3. Visualize Pred Labels
        label, pos = result['sem_pred'].reshape(-1,), result['occ_loc'].reshape(-1, 3)
        x = xx[pos[:, 0], pos[:, 1], pos[:, 2]]
        y = yy[pos[:, 0], pos[:, 1], pos[:, 2]]
        z = zz[pos[:, 0], pos[:, 1], pos[:, 2]]
        pred_sem_img = self.draw_voxels_3D(
            x, y, z,
            label,
            0.4,
        )
        
        # 4. Concat all images.
        row_1 = np.hstack(resize_imgs[:3])
        row_2 = np.hstack([gt_sem_image, pred_sem_img])
        
        if add_legend:
            legend = self.create_color_legend()
            H, W, _ = legend.shape
            ratio = self.W * 2 / W
            H1, W1 = int(H * ratio), int(W * ratio)
            legend = cv2.resize(legend, (W1, H1))
            row_2[30:H1+30] = legend
            
        row_3 = np.hstack(resize_imgs[3:])
        concat_img = np.vstack([row_1, row_2, row_3])
        cv2.imwrite(os.path.join(save_dir, f'{self.counter:0>6}_{scene_name}_{sample_token}.jpg'), concat_img)
        self.counter += 1
        