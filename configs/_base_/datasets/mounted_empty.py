
data_root = "/data/dataset"
dataset_type = "MountedEmpty"
#model_image_size = (1024, 1024) # for SAM
model_image_size = (512, 512)
keep_ratio = False
reduce_zero_label=False

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsFromCache'),
    dict(type='Resize', scale=model_image_size, keep_ratio=keep_ratio),
    dict(type='RandomFlip', prob=0.5, direction = "horizontal"),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=model_image_size, keep_ratio=keep_ratio),
    dict(type='LoadAnnotationsFromCache'),
    dict(type='PackSegInputs')
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=model_image_size , keep_ratio=keep_ratio)
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
            ],
            [dict(type='LoadAnnotationsFromCache')],
            [dict(type='PackSegInputs')]
        ])
]

# Dataset kwargs shared across splits.
# All images live in a single flat folder — no per-split copies needed.
# seg_map_path is intentionally absent: masks come from ann_npz_file, not individual PNGs.
_dataset_common = dict(
    type=dataset_type,
    data_root=data_root,
    reduce_zero_label=reduce_zero_label,
    data_prefix=dict(img_path='images'),
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        **_dataset_common,
        ann_npz_file='annotations/train.npz',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        **_dataset_common,
        ann_npz_file='annotations/val.npz',
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        **_dataset_common,
        ann_npz_file='annotations/test.npz',
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'], prefix="val")
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'], prefix="test")

