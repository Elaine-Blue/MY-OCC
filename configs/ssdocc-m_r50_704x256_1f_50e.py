dataset_type = 'NuScenesOccDataset'
dataset_root = 'data/nuscenes/'
occ_root = 'data/nuscenes/gts/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True
)

# For nuScenes we usually do 10-class detection
object_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]

occ_names_region = [
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]

occ_names_object = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
]

target_gt_labels_region = [
    occ_names.index(class_name) for class_name in occ_names_region
]

target_gt_labels_object = [
    occ_names.index(class_name) for class_name in occ_names_object
]

point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size = [0.4, 0.4, 0.4]

# arch config
embed_dims = 256
num_query_region = 2000
num_query_object = 400
num_query = num_query_region + num_query_object
num_refines_region = [1, 16, 32]
num_refines_object = [1, 2, 4, 8, 16, 32]
num_layers_region = len(num_refines_region)
num_layers_object = len(num_refines_object)
num_frames = 2
num_levels = 4
num_levels_region = 2
num_levels_object = 2
num_points = 2


img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=True)
img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels)
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True)

model = dict(
    type='SSDOCC',
    use_grid_mask=False,
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    stop_prev_grad=0,
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_bbox_head=dict(
        type='SSDOCCHead',
        occ_names=occ_names,
        num_classes=len(occ_names),
        branch_loss_weights=[0.3, 0.7, 0.4, 0.6], 
        region_aware_branch=dict(
            type='RegionAwareHead',
            num_classes=len(occ_names_region),
            in_channels=embed_dims,
            num_query=num_query_region,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            target_classes=target_gt_labels_region,
            use_priori_points=True,
            clusters_data_path='./cluster-centers_17-classes_5_clusters.pkl',
            transformer=dict(
                type='SSDTransformerDecoder',
                embed_dims=embed_dims,
                num_frames=num_frames,
                num_points=num_points,
                num_layers=num_layers_region,
                num_levels=num_levels_region,
                num_classes=len(occ_names_region),
                num_refines=num_refines_region,
                scales=[0.5],
                pc_range=point_cloud_range)
        ),
        object_aware_branch=dict(
            type='ObjectAwareHead',
            num_classes=len(occ_names_object),
            in_channels=embed_dims,
            num_query=num_query_object,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            target_classes=target_gt_labels_object,
            transformer=dict(
                type='SSDTransformerDecoder',
                embed_dims=embed_dims,
                num_frames=num_frames,
                num_points=num_points,
                num_layers=num_layers_object,
                num_levels=num_levels_object,
                num_classes=len(occ_names_object),
                num_refines=num_refines_object,
                scales=[0.5],
                pc_range=point_cloud_range)
            ),
        # TODO: Rewrite the following two loss to support multiple branches
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_pts=dict(type='SmoothL1Loss', beta=0.2, loss_weight=0.5)),
    train_cfg=dict(
        pts=dict(
            cls_weights=[
                10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5,  1, 5, 1, 1, 2, 1
            ],
            cls_weights_object=[
                10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5],
            cls_weights_region=[
                1, 5, 1, 1, 2, 1]
            )
        ),
    test_cfg=dict(
        pts=dict(
            score_thr=0.5,
            padding=True
        )
    )
)

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, ), # with_bbox=True, with_label=True, with_bbox_depth=True
    dict(type='LoadOccFromFile', occ_root=occ_root), 
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=object_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='DefaultFormatBundle3D', class_names=object_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'mask_camera',], meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape', 'ego2occ', 'ego2img', 
        'ego2lidar', 'img_timestamp', 'scene_name', 'sample_idx')) #  'gt_bboxes', 'gt_labels', 'centers2d', 'depths'
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='LoadOccFromFile', occ_root=occ_root), 
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=object_names, with_label=False),
            dict(type='Collect3D', keys=['img', 'voxel_semantics', 'mask_camera'], meta_keys=(
                'filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape',
                'ego2occ', 'ego2img', 'ego2lidar', 'img_timestamp', 'scene_name', 'sample_idx'))
        ])
]

data = dict(
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_train_mini_sweep.pkl',
        pipeline=train_pipeline,
        classes=object_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_test_sweep.pkl',
        pipeline=test_pipeline,
        classes=object_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01
)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2)
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)
total_epochs = 50
batch_size = 8

# load pretrained weights
load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]

# resume the last training
resume_from = None

# checkpointing
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

# logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', interval=50, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True),
        # dict(type='MEGVIIEMAHook2', init_updates=0, decay=0.999, resume=None),
        dict(type='VisualizationHook', interval=1000)
    ]
)

# evaluation
eval_config = dict(interval=total_epochs)

debug=False
