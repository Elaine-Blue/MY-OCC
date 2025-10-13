import sys
from pathlib import Path
project_path = Path(__file__).parent.parent
sys.path.append(str(project_path))
import cv2
import argparse
import importlib
import os
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from datetime import datetime
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d_plugin.loaders.builder import build_dataloader
from mmdet3d_plugin.hooks import visualize_results



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--model-weight-path', required=True, help='Path to checkpoint')
    parser.add_argument('--save-dir', type=str, default='visualizations', help='Visualize results')
    
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)
    
    work_dir = os.path.join(args.save_dir, 'single_sample_visualization')
    os.makedirs(work_dir, exist_ok=True)

    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    for p in cfgs.data.val.pipeline:
        if p['type'] == 'LoadMultiViewImageFromMultiSweeps':
            p['force_offline'] = True
    val_dataset = build_dataset(cfgs.data.val)
    
    val_dataset.data_infos = val_dataset.data_infos[:1]
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    model_weight_path = args.model_weight_path
    if model_weight_path.endswith('.pth'):
        model_weight_lst = [model_weight_path]
    else:
        model_weight_lst = [os.path.join(model_weight_path, file_name) for file_name in os.listdir(save_dir) if file_name.endswith('.pth')]
    
    model = build_model(cfgs.model)
    model.cuda()
    model = MMDataParallel(model, [0])
    
    for model_path in model_weight_lst:
        load_checkpoint(model, model_path, map_location='cuda', strict=False)
        model.eval()
        epoch = model_path.split('/')[-1].split('.')[0].split('_')[-1]
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                model(return_loss=False, **data)
                result_list = model.module.intermediate_results['result_list']
                
                matched_results = None
                
                pc_range = model.module.pts_bbox_head.pc_range.cpu().numpy()
                voxel_size = model.module.pts_bbox_head.voxel_size.cpu().numpy()
               
                visualize_results(
                    pc_range,
                    voxel_size,
                    result_list,
                    matched_results,
                    data,
                    work_dir,
                    epoch=epoch
                )
