
# Inherit and overwrite part of the config based on this config
base_dir = '/home/anuki/ICIP2022-ParEggChallenge'
_base_ = ['/home/anuki/ICIP2022-ParEggChallenge/configs/_base_/models/tood_r50_fpn.py']

data_root = '/home/anuki/ICIP2022-ParEggChallenge/sliced_data/sliced_data/cut_960x1280' # dataset root
split_root = '/home/anuki/ICIP2022-ParEggChallenge/5-fold/fold-0/'

train_batch_size_per_gpu = 4
train_num_workers = 2

max_epochs = 20

classes = ('Ascaris lumbricoides', 'Capillaria philippinensis', 'Enterobius vermicularis', 'Fasciolopsis buski',
           'Hookworm egg', 'Hymenolepis diminuta', 'Hymenolepis nana', 'Opisthorchis viverrine',
           'Paragonimus spp', 'Taenia spp. egg', 'Trichuris trichiura')

metainfo = {
    'classes': classes
}

dataset_type = 'CocoDataset'

img_norm_cfg = dict(
    mean=[126.09942512, 126.18320368, 111.28689804], std=[62.71579381, 60.8892706, 52.85799401], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280,960), keep_ratio=True), #(height, width)
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280,960),#(1060, 800),#(1224,735),#
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=split_root + 'cut_960x1280_train.json',
        img_prefix='',
        metainfo=metainfo,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=split_root + 'cut_960x1280_val.json',
        img_prefix='',
        metainfo=metainfo,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=split_root + 'cut_960x1280_val.json',
        img_prefix='',
        metainfo=metainfo,
        pipeline=test_pipeline))
evaluation = dict(metric='bbox', interval=4)

model = dict(bbox_head=dict(num_classes=len(classes)))

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
#visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])

seed = 0

gpu_ids = [0, 1]

load_from = None#'/home/anuki/ICIP2022-ParEggChallenge/checkpoints/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth'

work_dir = '/home/anuki/ICIP2022-ParEggChallenge/work_dirs/tood'
