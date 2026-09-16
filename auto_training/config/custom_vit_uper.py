custom_hooks = [
    dict(type='ForceRunTestLoop'),
]
data_root = '/dataset'
dataset_type = 'CustomMountedEmpty'
default_scope = 'mmseg'
keep_ratio = False
launcher = 'none'
load_from = None
log_level = 'INFO'
reduce_zero_label = False
resume = False
work_dir = '/results'
model_image_size = (512, 512)
norm_cfg = dict(requires_grad=True, type='SyncBN')

# <<< CHANGE: the only value to change between runs.
#     Run A (baseline): 0.0     Run B (with rotation): 0.75
ROT_PROB = 0.75

data_preprocessor = dict(
    bgr_to_rgb=False,
    mean=[123.675, 116.28, 103.53],
    pad_val=0,
    rgb_to_bgr=False,
    seg_pad_val=255,
    std=[58.395, 57.12, 57.375],
    type='SegDataPreProcessor')

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        load_best_for_testing='val/mDice',
        rule='greater',
        save_best='val/mDice',
        save_last=False,
        type='TestBestModelCheckpointHook'),
    logger=dict(interval=20, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=False, interval=1, modes=['test'], type='SegVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

log_processor = dict(by_epoch=False)

model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/deit_small_patch16_224-cd65a155.pth',
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[123.675, 116.28, 103.53],
        pad_val=0,
        rgb_to_bgr=False,
        seg_pad_val=255,
        size=(512, 512),
        std=[58.395, 57.12, 57.375],
        type='SegDataPreProcessor'),
    backbone=dict(
        act_cfg=dict(type='GELU'),
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=384,
        img_size=(512, 512),
        in_channels=3,
        interpolate_mode='bicubic',
        mlp_ratio=4,
        norm_cfg=dict(eps=1e-06, type='LN'),
        norm_eval=False,
        num_heads=6,
        num_layers=12,
        out_indices=(2, 5, 8, 11),
        patch_size=16,
        qkv_bias=True,
        type='VisionTransformer',
        with_cls_token=True),
    neck=None,
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        out_channels=2,
        pool_scales=(1, 2, 3, 6),
        type='UPerHead'),
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=384,
        in_index=3,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=1,
        out_channels=2,
        type='FCNHead'),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    optimizer=dict(
        betas=(0.9, 0.999), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            cls_token=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(
    betas=(0.9, 0.999), lr=6e-05, momentum=0.9, type='AdamW',
    weight_decay=0.01)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=10, start_factor=1e-06, type='LinearLR'),
    dict(begin=10, by_epoch=False, end=100, eta_min=0.0, power=0.9,
         type='PolyLR'),
]
train_cfg = dict(max_iters=100, type='IterBasedTrainLoop', val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ------------------------------------------------------------
# <<< CHANGE: train_pipeline is defined BEFORE train_dataloader
#     and the dataloader now actually uses it.
# ------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsFromCache'),
    dict(keep_ratio=False, scale=(512, 512), type='Resize'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='RandomRotate90', prob=ROT_PROB, debug_save_n=10),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images'),
        ann_npz_file='annotations/train.npz',
        reduce_zero_label=False,
        pipeline=train_pipeline))  # <<< CHANGE (previously a separate list without rotation)


# ------------------------------------------------------------
# <<< CHANGE: val and test = 4 copies (0/90/180/270)
# ------------------------------------------------------------
def _rot_pipeline(k):
    return [
        dict(type='LoadImageFromFile'),
        dict(type='FixedRot90', k=k, target='img'),
        dict(keep_ratio=False, scale=(512, 512), type='Resize'),
        dict(type='LoadAnnotationsFromCache'),
        dict(type='FixedRot90', k=k, target='seg'),
        dict(type='PackSegInputs',
             meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                        'pad_shape', 'scale_factor', 'flip',
                        'flip_direction', 'reduce_zero_label', 'rot_k')),
    ]


def _rot_dataset(npz):
    return dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(img_path='images'),
                ann_npz_file=npz,
                reduce_zero_label=False,
                pipeline=_rot_pipeline(k)) for k in range(4)
        ])


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(512, 512), type='Resize'),
    dict(type='LoadAnnotationsFromCache'),
    dict(type='PackSegInputs'),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=_rot_dataset('annotations/val.npz'))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images'),
        ann_npz_file='annotations/test.npz',
        reduce_zero_label=False,
        pipeline=test_pipeline))

val_evaluator = dict(
    type='RotationIoUMetric', iou_metrics=['mIoU', 'mDice'], prefix='val')
test_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice'], prefix='test')

tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [dict(keep_ratio=False, scale=(512, 512), type='Resize')],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [dict(type='LoadAnnotationsFromCache')],
            [dict(type='PackSegInputs')],
        ],
        type='TestTimeAug'),
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    alpha=0.7,
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])