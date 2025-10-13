import os
import cv2
import os.path as osp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mmcv.runner import HOOKS, Hook
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
import io

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
    17: (140, 140, 140),  # Green vegetation
}

occ_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]

palette = np.array([classname_to_color[i] for i in range(len(classname_to_color))])
color_mapping = {name: color for name, color in zip(occ_names, classname_to_color.values())}

RESIZE_SAHPE = (640, 480)
H = W = RESIZE_SAHPE[0] * 1.5
SL, LW, TL = 20, 20, 180

def world_to_voxel(points, voxel_size=[0.4, 0.4, 0.4], pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]):
    points = points.copy()
    points[..., 0] = np.clip(points[..., 0], pc_range[0], pc_range[3] - voxel_size[0])
    points[..., 1] = np.clip(points[..., 1], pc_range[1], pc_range[4] - voxel_size[1])
    points[..., 2] = np.clip(points[..., 2], pc_range[2], pc_range[5] - voxel_size[2])
    
    points[..., 0] = (points[..., 0] - pc_range[0]) / voxel_size[0]
    points[..., 1] = (points[..., 1] - pc_range[1]) / voxel_size[1]
    points[..., 2] = (points[..., 2] - pc_range[2]) / voxel_size[2]
    
    return points.astype(np.int32)

def create_color_legend():
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
            font = ImageFont.truetype('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf', 20)
        else:
            font = ImageFont.truetype('arial.ttf', 20)
        draw.rectangle([start_x_cube, start_y_cube, start_x_cube+SL, start_y_cube+LW], fill=color)
        draw.text([start_x_cube+SL+5, start_y_cube], name, fill=(0, 0, 0), font=font)
    return np.array(legend)

def visualize_occ(x, y, z, labels, palette, voxel_size):
    """
    使用 Plotly 实现 3D 体素可视化
    
    参数:
    x, y, z: 坐标数组，表示体素在3D空间中的位置
    labels: 类别标签数组
    palette: 调色板，形状为 (n_classes, 3) 或 (n_classes, 4)
    voxel_size: 体素大小
    classes: 类别列表
    mode: 保留参数（Plotly 中不使用）
    color: 保留参数（Plotly 中使用 palette）
    show: 是否显示图像
    """
    
    if palette.max() > 1.0:
        palette = palette.astype(np.float32) / 255.0
    
    colors = []
    valid_indices = []
        
    for i, label in enumerate(labels):
        if 0 < label < len(palette):
            colors.append(palette[int(label)])
            valid_indices.append(i)

    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
    z_valid = z[valid_indices]
    colors = np.array(colors)
        
    fig = go.Figure()
    
    scatter = go.Scatter3d(
        x=x_valid,
        y=y_valid,
        z=z_valid,
        mode='markers',
        marker=dict(
            size=max(voxel_size, 1.0),
            color=colors,
            opacity=0.9,
            symbol='square',
        ),
        hoverinfo='skip',
        showlegend=False
    )
    
    fig.add_trace(scatter)
    
    if len(z) > 0:
        z_min, z_max = z.min(), z.max()
        z_ticks = np.linspace(z_min, z_max, 2)
    else:
        z_ticks = [0]
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', backgroundcolor="white", gridcolor="lightgray"),
            yaxis=dict(title='Y', backgroundcolor="white", gridcolor="lightgray"),
            zaxis=dict(
                title='Z', 
                backgroundcolor="white", 
                gridcolor="white",
                tickvals=z_ticks,
                ticktext=[f"{val:.1f}" for val in z_ticks]
            ),
            bgcolor='white',
            aspectmode='data'
        ),
        width=W,
        height=H,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='white',
    )
    
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=2.5, y=2.5, z=2.5),
            up=dict(x=0, y=0, z=5)
        )
    )
    img_bytes = fig.to_image(format="png", width=W, height=H)
    img = Image.open(io.BytesIO(img_bytes))
    save_fig = np.array(img)[:, :, :3]
    
    legend = create_color_legend()
    lw, lh = legend.shape[1], legend.shape[0]
    save_fig[10:10+lh, 10:10+lw] = legend
    return save_fig

def draw_matched_results(matched_results, seq_idx, save_path):
    gt_paired_pts = matched_results['gt_paired_pts'][seq_idx][::1000].detach().cpu().numpy()
    pred_paired_pts = matched_results['pred_paired_pts'][seq_idx][::1000].detach().cpu().numpy()
    gt_paired_labels = matched_results['gt_paired_labels'][seq_idx][::1000].detach().cpu().numpy()
    pred_paired_labels = matched_results['pred_paired_labels'][seq_idx][::1000].detach().cpu().numpy()
    assert len(gt_paired_pts) == len(pred_paired_pts) == len(gt_paired_labels) == len(pred_paired_labels)
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(gt_paired_pts)):
        gt_point = gt_paired_pts[i]
        pred_point = pred_paired_pts[i]
        
        gt_label = gt_paired_labels[i]
        pred_label = pred_paired_labels[i]
        
        gt_color = classname_to_color.get(gt_label, 'black')
        gt_color = (np.array(gt_color) / 255.0).astype('float32')
        pred_color = classname_to_color.get(pred_label, 'black')
        pred_color = (np.array(pred_color) / 255.0).astype('float32')
        
        ax.scatter(gt_point[0], gt_point[1], gt_point[2], 
                  color=gt_color, marker='o', s=20, label=occ_names[gt_label])
        
        ax.scatter(pred_point[0], pred_point[1], pred_point[2], 
                  color=pred_color, marker='*', s=20, label=occ_names[pred_label])
        
        ax.plot([gt_point[0], pred_point[0]], 
               [gt_point[1], pred_point[1]], 
               [gt_point[2], pred_point[2]], 
               color='gray', linestyle='-', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_title('3D Voxel Pair Matching Visualization')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    save_fig = cv2.imread(save_path)
    legend = create_color_legend()
    lw, lh = legend.shape[1], legend.shape[0]
    save_fig[10:10+lh, 10:10+lw] = legend
    cv2.imwrite(save_path, save_fig)

def visualize_results(pc_range, voxel_size, result_list, matched_results, data_batch, save_dir, epoch, mode='cube'):
    W = int((pc_range[3] - pc_range[0]) / voxel_size[0])
    H = int((pc_range[4] - pc_range[1]) / voxel_size[1])
    Z = int((pc_range[5] - pc_range[2]) / voxel_size[2])

    x = (np.arange(0, W) + 0.5) * voxel_size[0] + pc_range[0]
    y = (np.arange(0, H) + 0.5) * voxel_size[1] + pc_range[1]
    z = (np.arange(0, Z) + 0.5) * voxel_size[2] + pc_range[2]
    xx = x[:, None, None].repeat(H, axis=1).repeat(Z, axis=2)
    yy = y[None, :, None].repeat(W, axis=0).repeat(Z, axis=2)
    zz = z[None, None, :].repeat(W, axis=0).repeat(H, axis=1)
    for i, result in enumerate(result_list):
        img_metas = data_batch['img_metas'].data[0][i]
        sample_token = img_metas['sample_idx']
        # 1. Visualize Camera Images.
        ori_imgs = img_metas['ori_imgs']
        resize_imgs = []
        camera_names = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        for camera_name, img in zip(camera_names, ori_imgs):
            img = img.copy()
            img = cv2.resize(img, RESIZE_SAHPE)
            cv2.putText(img, camera_name, (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            resize_imgs.append(img)
        
        H, W = resize_imgs[0].shape[:2]
        # 2. Visualize GT Labels
        occ = data_batch['voxel_semantics'][i].cpu().numpy()
        x, y, z = xx[occ!=17], yy[occ!=17], zz[occ!=17]
        label = occ[occ!=17].astype(np.int64)
        gt_sem_image = visualize_occ(
            x, y, z,
            label,
            palette,
            0.4)
        
        # 3. Visualize Pred Labels
        label, pos = result['sem_pred'], result['occ_loc']
        x = xx[pos[:, 0], pos[:, 1], pos[:, 2]]
        y = yy[pos[:, 0], pos[:, 1], pos[:, 2]]
        z = zz[pos[:, 0], pos[:, 1], pos[:, 2]]
        pred_sem_img = visualize_occ(
            x, y, z,
            label,
            palette,
            0.4)
        # 4. Concat all images.
        row_1 = np.hstack(resize_imgs[:3])
        row_2 = np.hstack([gt_sem_image, pred_sem_img])
        row_3 = np.hstack(resize_imgs[3:])
        concat_img = np.vstack([row_1, row_2, row_3])
        cv2.imwrite(osp.join(save_dir, f'epoch_{epoch:0>3}_{i:0>6}_{sample_token}.jpg'), concat_img)

        # Visualize 3D Voxel Pair Matching.
        save_img_path = osp.join(save_dir, f'epoch_{epoch:0>3}_{i:0>6}_{sample_token}_matched_pair.jpg')
        draw_matched_results(matched_results, i, save_img_path)
        
@HOOKS.register_module()
class VisualizationHook(Hook):
    """
    Function to visualize results during training.
    
    Args:
        interval (int): Visualization interval.
        out_dir (str): Visualize results save directory.
        score_thr (float): Visualization score threshold.
    """
    
    def __init__(self, interval=1000):
        self.interval = interval
    
    def after_train_iter(self, runner):
        """Visualize results after each training iteration."""
        if self.every_n_iters(runner, self.interval):
            model = runner.model.module
            result_list = model.intermediate_results['result_list']
            matched_results = model.intermediate_results['matched_results']
            result_list[0]['sem_pred'] = matched_results['pred_paired_labels'][0].detach().cpu().numpy()
            result_list[0]['occ_loc'] = world_to_voxel(matched_results['pred_paired_pts'][0].detach().cpu().numpy())    
            pc_range = model.pts_bbox_head.pc_range.cpu().numpy()
            voxel_size = model.pts_bbox_head.voxel_size.cpu().numpy()
            data_batch = runner.data_batch
            epoch = runner.epoch
            save_dir = osp.join(runner.work_dir, 'train_visiualization')
            os.makedirs(save_dir, exist_ok=True)
            visualize_results(
                pc_range,
                voxel_size,
                result_list,
                matched_results,
                data_batch,
                save_dir,
                epoch
            )

