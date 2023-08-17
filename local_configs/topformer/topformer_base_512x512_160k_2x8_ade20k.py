_base_ = [
    '../_base_/datasets/LPCVC_train_val.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py',
    './topformer_base.py'
]

optimizer = dict(
    type='AdamW',
    lr=0.0003,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=32)
find_unused_parameters=True

runner = dict(type='IterBasedRunner', max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=2500)
evaluation = dict(interval=2500, metric='mIoU', pre_eval=True, save_best = 'mIoU')
