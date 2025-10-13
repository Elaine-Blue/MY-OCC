import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
import io, os
import cv2

classname_to_color_cv = {  # RGB.
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
    17: (140, 140, 140),  # Green vegetation
}

occ_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]

color_map = {
    key: (color[2]/255.0, color[1]/255.0, color[0]/255.0)
    for key, color in classname_to_color_cv.items()
}

class Visualizer:
    """3D voxel Matched Results Visualizer"""
    def __init__(self, figsize=(16, 6)):
        self.figsize = figsize
        self.color_map = color_map
        self.class_names = occ_names
    
    
    def create_color_legend(self, SL=20, LW=20, TL=180, font_size=20):
        color_mapping = {name: color for name, color in zip(occ_names, classname_to_color_cv.values())}
        cmap = color_mapping.copy()
        cmap.pop('others')
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
        return np.array(legend)

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

    def draw_voxels_3D(self, x, y, z, labels, colors, voxel_size, pc_range):
        """
        Draw Voxels in 3D space using Plotly.
        
        Parameters:
            - x, y, z: (n, ) world coordinates of voxels
            - labels: (n, ) voxel semantic labels
            - palette: (n, 3) color palette for each semantic label
            - voxel_size: Float
            - pc_range: List of point cloud range
        """
        
        fig = go.Figure()
        
        # Draw Background Voxels First
        unique_labels = np.unique(labels)[::-1]
        
        for label in unique_labels:
            mask = labels == label
            if not np.any(mask):
                continue
                
            x_class = x[mask]
            y_class = y[mask] 
            z_class = z[mask]
            
            color = colors[int(label)]
            color_rgba = f'rgba({int(color[2])}, {int(color[1])}, {int(color[0])}, 0.6)'
            
            all_vertices = []
            all_faces = []
            face_offset = 0
            
            for i in range(len(x_class)):
                center = [x_class[i], y_class[i], z_class[i]]
                vertices, faces = self.create_voxel_cube(center, voxel_size)
                all_vertices.append(vertices)
                adjusted_faces = faces + face_offset
                all_faces.append(adjusted_faces)
                face_offset += 8 
            
            # Merge all vertices and faces
            if all_vertices:
                all_vertices = np.vstack(all_vertices)
                all_faces = np.vstack(all_faces)
                
                # Add Mesh3d trace
                fig.add_trace(go.Mesh3d(
                    x=all_vertices[:, 0],
                    y=all_vertices[:, 1],
                    z=all_vertices[:, 2],
                    i=all_faces[:, 0],
                    j=all_faces[:, 1],
                    k=all_faces[:, 2],
                    color=color_rgba,
                    flatshading=True,
                    name=f'{occ_names[label]}',
                    showlegend=True
                ))
        
        if len(z) > 0:
            z_min, z_max = z.min(), z.max()
            z_ticks = [round((z_min + z_max)/2, 1), z_max]
        else:
            z_ticks = [5.0]
            
        # Update layout
        fig.update_layout(
            title=dict(
                text='3D Voxels Visualization',
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(
                    family='Times New Roman',
                    size=30,
                    color='black',
                    weight='bold',
                ),
            ),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)', 
                zaxis_title='Z (m)',
                xaxis=dict(
                    range=[pc_range[0], pc_range[3]],
                    backgroundcolor='white',
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    range=[pc_range[1], pc_range[4]],
                    backgroundcolor='white',
                    gridcolor='lightgray'
                ),
                zaxis=dict(
                    range=[pc_range[2], pc_range[5]],
                    backgroundcolor='white',
                    gridcolor='lightgray',
                    tickvals=z_ticks,
                    ticktext=[f"{val:.1f}" for val in z_ticks]
                ),
                bgcolor='white',
                aspectmode='data'
            ),
            legend=dict(
                orientation='h',    # horizontal
                y=0.90,             # top
                x=0.5,              # center
                xanchor='center',
                yanchor='top',
                bgcolor='white',
                bordercolor='black',
                font=dict(
                    family='Times New Roman',
                    size=25,
                    color='black',
                ),
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Adjust camera view
        camera_view = dict(
                        eye=dict(x=2.0, y=2.0, z=1.2),
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0)
        )
        
        fig.update_layout(scene_camera=camera_view)
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
        img = Image.open(io.BytesIO(img_bytes))
        save_fig = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
        return save_fig