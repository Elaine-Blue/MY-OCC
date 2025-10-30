import sys
from pathlib import Path
project_path = Path(__file__).parent.parent
sys.path.append(str(project_path))

import os
import cv2
import os.path as osp
import numpy as np
from mmcv.runner import HOOKS, Hook
from mmdet3d_plugin.models.bbox.utils import decode_points
from tools.visualizer import Visualizer, world_to_voxel 

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
    18: (0, 255, 255) # init points
}


occ_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]

palette = np.array([classname_to_color[i] for i in range(len(classname_to_color))])


RESIZE_SAHPE = (1600, 1200)
H = W = RESIZE_SAHPE[0] * 1.5

def draw_matched_results(vis_tool, matched_results, seq_idx, save_path, max_num_points=500):
    gt_paired_pts = matched_results['gt_paired_pts'][seq_idx].detach().cpu().numpy()
    pred_paired_pts = matched_results['pred_paired_pts'][seq_idx].detach().cpu().numpy()
    gt_paired_labels = matched_results['gt_paired_labels'][seq_idx].detach().cpu().numpy()
    pred_paired_labels = matched_results['pred_paired_labels'][seq_idx].detach().cpu().numpy()
    
    factor = (len(gt_paired_pts) // max_num_points) + 1
    img_rgba = vis_tool.draw_matched_results(pred_paired_pts, gt_paired_pts, pred_paired_labels, gt_paired_labels, factor)
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

    cv2.imwrite(save_path, img_bgr)

def visualize_results(pc_range, voxel_size, results, matched_results, data_batch, save_dir, epoch, sample_idx):
    W = int((pc_range[3] - pc_range[0]) / voxel_size[0])
    H = int((pc_range[4] - pc_range[1]) / voxel_size[1])
    Z = int((pc_range[5] - pc_range[2]) / voxel_size[2])

    x = (np.arange(0, W) + 0.5) * voxel_size[0] + pc_range[0]
    y = (np.arange(0, H) + 0.5) * voxel_size[1] + pc_range[1]
    z = (np.arange(0, Z) + 0.5) * voxel_size[2] + pc_range[2]
    xx = x[:, None, None].repeat(H, axis=1).repeat(Z, axis=2)
    yy = y[None, :, None].repeat(W, axis=0).repeat(Z, axis=2)
    zz = z[None, None, :].repeat(W, axis=0).repeat(H, axis=1)
    
    B = len(data_batch['img_metas'].data[0])
    
    vis_tool = Visualizer()
    
    for i in range(B):
        img_metas = data_batch['img_metas'].data[0][i]
        scene_name = img_metas['scene_name']
        sample_token = img_metas['sample_idx']
        
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
            img = cv2.resize(img, RESIZE_SAHPE)
            cv2.putText(img, camera_name, (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            resize_imgs.append(img)
        
        '''
        init_points = results.get('init_points', None)
        if init_points is not None:
            init_points = init_points[i].reshape(-1, 3)
            init_world_coords = decode_points(init_points[:, :3], pc_range)
            init_points[:, :3] = init_world_coords
        '''
        
        # 2. Visualize GT Labels
        occ_labels = data_batch['voxel_semantics'][i].cpu().numpy()
        x, y, z = xx[occ_labels!=17], yy[occ_labels!=17], zz[occ_labels!=17]
        label = occ_labels[occ_labels!=17].astype(np.int64)
        gt_sem_image = vis_tool.draw_voxels_3D(
            x, y, z,
            label,
            palette,
            0.4,
            pc_range,
        )
        
        # 3. Visualize Pred Labels
        label, pos = results['sem_pred'][i].reshape(-1,), results['occ_loc'][i].reshape(-1, 3)
        x = xx[pos[:, 0], pos[:, 1], pos[:, 2]]
        y = yy[pos[:, 0], pos[:, 1], pos[:, 2]]
        z = zz[pos[:, 0], pos[:, 1], pos[:, 2]]
        pred_sem_img = vis_tool.draw_voxels_3D(
            x, y, z,
            label,
            palette,
            0.4,
            pc_range
        )
        
        # 4. Concat all images.
        row_1 = np.hstack(resize_imgs[:3])
        row_2 = np.hstack([gt_sem_image, pred_sem_img])
        row_3 = np.hstack(resize_imgs[3:])
        concat_img = np.vstack([row_1, row_2, row_3])
        cv2.imwrite(osp.join(save_dir, f'{sample_idx:0>6}_epoch_{epoch:0>3}_{i:0>3}_{scene_name}_{sample_token}.jpg'), concat_img)

        # Visualize 3D Voxel Pair Matching.
        if matched_results:
            save_img_path = osp.join(save_dir, f'{sample_idx:0>6}_epoch_{epoch:0>3}_{i:0>3}_{scene_name}_{sample_token}_matched_pairs.jpg')
            draw_matched_results(vis_tool, matched_results, i, save_img_path)  

def post_process(result_list, pc_range, voxel_size):
    # Convert all results to numpy.array.
    result_list['init_points'] = result_list['init_points'].detach().cpu().numpy()
    all_refine_pts = result_list['all_refine_pts']
    refine_pts = all_refine_pts[-1]
    all_cls_scores = result_list['all_cls_scores']
    cls_scores = all_cls_scores[-1].sigmoid()
    sem_labels = cls_scores.max(-1)[1].detach().cpu().numpy()
    result_list['sem_pred'] = sem_labels
    
    norm_coords = refine_pts.detach().cpu().numpy()
    world_coords = decode_points(norm_coords, pc_range)
    voxel_coords = world_to_voxel(world_coords, voxel_size, pc_range)
    result_list['occ_loc'] = voxel_coords
    return result_list

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
            
            # Get dt-voxels & gt-voxels matched results
            matched_results = model.intermediate_results.get('matched_results', {})
            
            # Post-process results
            pc_range = model.pts_bbox_head.pc_range.cpu().numpy()
            voxel_size = model.pts_bbox_head.voxel_size.cpu().numpy()
            
            result_list = post_process(result_list, pc_range, voxel_size)
            
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
                epoch,
                self.interval
            )

