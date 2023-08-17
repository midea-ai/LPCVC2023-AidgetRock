dataset_type = 'LPCVCDataset'
data_root = '/data/private/TopFormer/data/ade/LPCVC_v3_train_val'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline0 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(1.0, 1.0)),
    dict(type='RandomFlip', prob=0.2),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
train_pipeline1 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 600), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.8),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
train_pipeline2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(600, 2048), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.8),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
train_pipeline3 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(600, 600), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip', prob=0.9),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
train0 = dict(
    type='RepeatDataset',
    times=3,
    dataset=dict(
        type='LPCVCDataset',
        data_root='/data/private/TopFormer/data/ade/LPCVC_v3_train_val',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(512, 512), ratio_range=(1.0, 1.0)),
            dict(type='RandomFlip', prob=0.2),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]))
train1 = dict(
    type='LPCVCDataset',
    data_root='/data/private/TopFormer/data/ade/LPCVC_v3_train_val',
    img_dir='images/training',
    ann_dir='annotations/training',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='Resize', img_scale=(2048, 600), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.8),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
train2 = dict(
    type='LPCVCDataset',
    data_root='/data/private/TopFormer/data/ade/LPCVC_v3_train_val',
    img_dir='images/training',
    ann_dir='annotations/training',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='Resize', img_scale=(600, 2048), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.8),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
train3 = dict(
    type='LPCVCDataset',
    data_root='/data/private/TopFormer/data/ade/LPCVC_v3_train_val',
    img_dir='images/training',
    ann_dir='annotations/training',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='Resize', img_scale=(600, 600), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='RepeatDataset',
                times=3,
                dataset=dict(
                    type='LPCVCDataset',
                    data_root=
                    '/data/private/TopFormer/data/ade/LPCVC_v3_train_val',
                    img_dir='images/training',
                    ann_dir='annotations/training',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', reduce_zero_label=False),
                        dict(
                            type='Resize',
                            img_scale=(512, 512),
                            ratio_range=(1.0, 1.0)),
                        dict(type='RandomFlip', prob=0.2),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(
                            type='Pad',
                            size=(512, 512),
                            pad_val=0,
                            seg_pad_val=255),
                        dict(type='DefaultFormatBundle'),
                        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
                    ])),
            dict(
                type='LPCVCDataset',
                data_root='/data/private/TopFormer/data/ade/LPCVC_v3_train_val',
                img_dir='images/training',
                ann_dir='annotations/training',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', reduce_zero_label=False),
                    dict(
                        type='Resize',
                        img_scale=(2048, 600),
                        ratio_range=(0.5, 2.0)),
                    dict(
                        type='RandomCrop',
                        crop_size=(512, 512),
                        cat_max_ratio=0.75),
                    dict(type='RandomFlip', prob=0.8),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(512, 512),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
                ]),
            dict(
                type='LPCVCDataset',
                data_root='/data/private/TopFormer/data/ade/LPCVC_v3_train_val',
                img_dir='images/training',
                ann_dir='annotations/training',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', reduce_zero_label=False),
                    dict(
                        type='Resize',
                        img_scale=(600, 2048),
                        ratio_range=(0.5, 2.0)),
                    dict(
                        type='RandomCrop',
                        crop_size=(512, 512),
                        cat_max_ratio=0.75),
                    dict(type='RandomFlip', prob=0.8),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(512, 512),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
                ]),
            dict(
                type='LPCVCDataset',
                data_root='/data/private/TopFormer/data/ade/LPCVC_v3_train_val',
                img_dir='images/training',
                ann_dir='annotations/training',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', reduce_zero_label=False),
                    dict(
                        type='Resize',
                        img_scale=(600, 600),
                        ratio_range=(0.5, 2.0)),
                    dict(
                        type='RandomCrop',
                        crop_size=(512, 512),
                        cat_max_ratio=0.75),
                    dict(type='RandomFlip', prob=0.5),
                    dict(type='PhotoMetricDistortion'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(512, 512),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
                ])
        ]),
    val=dict(
        type='LPCVCDataset',
        data_root='/data/private/TopFormer/data/ade/LPCVC_v3_train_val',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip', prob=0.9),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='LPCVCDataset',
        data_root='/data/private/TopFormer/data/ade/LPCVC_v3_train_val',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip', prob=0.9),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=0.0003,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=0.95,
    min_lr=0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(
    interval=5000, metric='mIoU', pre_eval=True, save_best='mIoU')
norm_cfg = dict(type='SyncBN', requires_grad=True)
model_cfgs = dict(
    cfg=[[3, 1, 16, 1], [3, 4, 32, 2], [3, 3, 32, 1], [5, 3, 64, 2],
         [5, 3, 64, 1], [3, 3, 128, 2], [3, 3, 128, 1], [5, 6, 160, 2],
         [5, 6, 160, 1], [3, 6, 160, 1]],
    channels=[32, 64, 128, 160],
    out_channels=[None, 256, 256, 256],
    embed_out_indice=[2, 4, 6, 9],
    decode_out_indices=[1, 2, 3],
    num_heads=8,
    c2t_stride=2)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='Topformer',
        cfgs=[[3, 1, 16, 1], [3, 4, 32, 2], [3, 3, 32, 1], [5, 3, 64, 2],
              [5, 3, 64, 1], [3, 3, 128, 2], [3, 3, 128, 1], [5, 6, 160, 2],
              [5, 6, 160, 1], [3, 6, 160, 1]],
        channels=[32, 64, 128, 160],
        out_channels=[None, 256, 256, 256],
        embed_out_indice=[2, 4, 6, 9],
        decode_out_indices=[1, 2, 3],
        depths=4,
        num_heads=8,
        c2t_stride=2,
        drop_path_rate=0.1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='modelzoos/classification/topformer-B-224-75.3.pth')),
    decode_head=dict(
        type='SimpleHead',
        in_channels=[256, 256, 256],
        in_index=[0, 1, 2],
        channels=256,
        dropout_ratio=0.1,
        num_classes=14,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0),
            dict(type='FocalLoss', loss_name='loss_focal', loss_weight=0.4),
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_entropy',
                use_sigmoid=False,
                loss_weight=1.7)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
find_unused_parameters = True
work_dir = 'teachers/base_0003_17ce_7-26'
gpu_ids = [0]
