
custom_imports = dict(
    imports=[
        'projects.mmdet3d_plugin',
        'projects.mmdet3d_plugin.datasets.nuscenes_dataset',
        'projects.mmdet3d_plugin.datasets.pipelines.transform_3d',
        'projects.mmdet3d_plugin.bevformer.modules.encoder',
        'projects.mmdet3d_plugin.bevformer.modules.transformer',
        'projects.mmdet3d_plugin.bevformer.modules.compat_registries',
        'projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder',
        'projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d',
        'projects.mmdet3d_plugin.core.bbox.match_costs',
        'projects.mmdet3d_plugin.bevformer.modules.ensure_time_hook',
        'mmdet3d.models',

        # PointPillars pieces
        'mmdet3d.models.voxel_encoders.pillar_encoder',   # PillarFeatureNet
        'mmdet3d.models.middle_encoders.pillar_scatter',  # PointPillarsScatter

        # Typical BEV backbone/neck used with PointPillars
        'mmdet3d.models.backbones.second',                # SECOND
        'mmdet3d.models.necks.second_fpn',                # SECONDFPN

        # (often needed) layers / ops wrappers
        'mmcv.ops',
    ],
    allow_failed_imports=False,
)

dist_params = dict(backend='gloo')
find_unused_parameters = True


_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

resume_from = None
auto_resume = False
log_level = 'INFO'

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'


# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]




img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 
    'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 100
bev_w_ = 100
queue_length = 3 # each sequence contains `queue_length` frames.

model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0,1,2,3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[256,512,1024,2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    # -------------------------
    # LiDAR branch: PointPillars
    # -------------------------
    pts_voxel_layer=dict(
        max_num_points=20,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,          # [0.2, 0.2, 8] -> pillar z is single bin
        max_voxels=(30000, 40000),      # (train, test) adjust if OOM / too sparse
    ),

    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,                  # x,y,z,intensity (common)
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
    ),

    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        # output_shape is (H, W) = (y_cells, x_cells)
        output_shape=(512, 512),        # matches your train_cfg.grid_size [512,512,1]
    ),

    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=300,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            num_cams=6,
            use_prev_bev=True,
            pc_range=point_cloud_range,
            num_feature_levels=4,
            fusion_mode='encoder',
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='MM_BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            batch_first=True,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=4,
                                batch_first=True,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                            num_cams=6,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    batch_first=True,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'),
                    lidar_cross_attn_layer=dict(
                        type='CustomMSDeformableAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        num_levels=1,
                        num_points=4,
                        batch_first=True,
                        dropout=0.1,
                    ),
                )),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
                #use_prev_bev=False,
              # keep temporal fully off for now 
        ),
        
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.25)), #0.0
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.25), # 0.0 Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))),
    test_cfg=dict(
        pts=dict(
            score_thr=0.05, #0.35,          # start around 0.15–0.3
            max_num=300, #150,        # avoid flooding JSON
            #nms_type='circle',      # or 'nms' depending on your repo
            #circle_radius=2.5,      # for center-distance NMS (if supported)
            # nms={'type': 'nms', 'iou_threshold': 0.2},  # if IoU NMS is used
        )
    )            
            
)

dataset_type = 'CustomNuScenesDataset'
data_root = r'/home/xiaotwu/Code/BEVFormerFusion/data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,             # common nuScenes: x,y,z,intensity,ring(or time)
        use_dim=4,              # use x,y,z,intensity
        file_client_args=file_client_args,
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),

    # Optional but usually recommended for LiDAR
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),

    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),

    dict(type='CustomCollect3D', 
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img'],
         )
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        file_client_args=file_client_args,
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
   
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['points', 'img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=1.0e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01,
    )

optimizer_config = dict(
    #type='Fp16OptimizerHook',
    grad_clip=dict(max_norm=1.0, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear', warmup_iters=2000, warmup_ratio=1.0/1000,
    #warmup_ratio=1.0 / 3,
    min_lr_ratio=0.1)
#total_epochs = 96

evaluation = dict(interval=2000, pipeline=test_pipeline)

runner = dict(type='IterBasedRunner', max_iters=8000)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [
    dict(type='IterTimerHook'),
    dict(type='EnsureTimeDataHook', priority='LOW'),
]

checkpoint_config = dict(interval=2000, by_epoch=False)
