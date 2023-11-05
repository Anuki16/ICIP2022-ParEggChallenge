
# Inherit and overwrite part of the config based on this config
_base_ = '/home/anuki/ICIP2022-ParEggChallenge/checkpoints/rtmdet_s_8xb32-300e_coco.py'

data_root = '/home/anuki/ICIP2022-ParEggChallenge/sliced_data/sliced_data/cut_960x1280' # dataset root
split_root = '/home/anuki/ICIP2022-ParEggChallenge/5-fold/fold-1/'

train_batch_size_per_gpu = 4
train_num_workers = 2

max_epochs = 20
stage2_num_epochs = 1
base_lr = 0.00008


metainfo = {
    'classes': ('Ascaris lumbricoides', 'Capillaria philippinensis', 'Enterobius vermicularis', 'Fasciolopsis buski',
           'Hookworm egg', 'Hymenolepis diminuta', 'Hymenolepis nana', 'Opisthorchis viverrine',
           'Paragonimus spp', 'Taenia spp. egg', 'Trichuris trichiura'),
    'palette': [
        (220, 20, 60),
        (255, 0, 0),
        (0, 255, 0),
        (0,0,255),
        (255,100,0),
        (100,255,0),
        (100,255,255),
        (255,100,100),
        (100,255,100),
        (100,100,255),
        (255,255,255),
    ]
}

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=''),
        ann_file=split_root + 'cut_960x1280_train.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=''),
        ann_file=split_root + 'cut_960x1280_val.json'))

test_dataloader = dict(
    dataset=dict(
        data_root='/home/anuki/ICIP2022-ParEggChallenge/test/data',
        metainfo=metainfo,
        data_prefix=dict(img=''),
        ann_file='/home/anuki/ICIP2022-ParEggChallenge/test/test_labels.json'))

val_evaluator = dict(ann_file=split_root + 'cut_960x1280_val.json')

test_evaluator = dict(ann_file='/home/anuki/ICIP2022-ParEggChallenge/test/test_labels.json')

model = dict(bbox_head=dict(num_classes=11))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=10),
    dict(
        # use cosine lr from 10 to 20 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs//2,
        end=max_epochs,
        T_max=0,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=2,  # only keep latest 2 checkpoints
        save_best='auto'
    ),
    logger=dict(type='LoggerHook', interval=5))

custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# load COCO pre-trained weight
load_from = '/home/anuki/ICIP2022-ParEggChallenge/checkpoints/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
