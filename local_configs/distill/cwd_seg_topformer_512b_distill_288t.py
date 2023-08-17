_base_ = [
    '../_base_/datasets/LPCVC_distill.py',       # data
    '../_base_/schedules/schedule_for_distill.py',    # training schedule
    '../_base_/default_runtime.py'                    # runtime setting
]

# specify norm_cfg for teacher and student as follows
norm_cfg = dict(type='SyncBN', requires_grad=True)

student_model_cfgs =dict(
    cfg=[
    # k,  t,  c, s
        [3,   1,  16, 1], # 1/2        0.464K  17.461M
        [3,   4,  16, 2], # 1/4 1      3.44K   64.878M
        [3,   3,  16, 1], #            4.44K   41.772M
        [5,   3,  32, 2], # 1/8 3      6.776K  29.146M
        [5,   3,  32, 1], #            13.16K  30.952M
        [3,   3,  64, 2], # 1/16 5     16.12K  18.369M
        [3,   3,  64, 1], #            41.68K  24.508M
        [5,   6,  96, 2], # 1/32 7     0.129M  36.385M
        [5,   6,  96, 1], #            0.335M  49.298M
    ],
    channels=[16, 32, 64, 96],
    out_channels=[None, 128, 128, 128],
    embed_out_indice=[2, 4, 6, 8],
    decode_out_indices=[1, 2, 3],
    num_heads=4,
    c2t_stride=2,
)

student_checkpoint='output/tiny_288_8tp_3trans_7-31/iter_62500.pth'
student = dict(
    type='mmseg.EncoderDecoder',
    init_cfg=dict(type='Pretrained', checkpoint=student_checkpoint),
    backbone=dict(
        type='Topformer',
        cfgs=student_model_cfgs['cfg'], 
        channels=student_model_cfgs['channels'],
        out_channels=student_model_cfgs['out_channels'], 
        embed_out_indice=student_model_cfgs['embed_out_indice'],
        decode_out_indices=student_model_cfgs['decode_out_indices'],
        depths=4,
        num_heads=student_model_cfgs['num_heads'],
        c2t_stride=student_model_cfgs['c2t_stride'],
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='SimpleHead',
        in_channels=[128, 128, 128],
        in_index=[0, 1, 2],
        channels=128,
        dropout_ratio=0.1,
        num_classes=14,
        is_dw=True,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(
            # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        loss_decode=[
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0),
            dict(type='FocalLoss',loss_name='loss_focal',loss_weight=0.3),
            dict(type='CrossEntropyLoss', loss_name='loss_entropy',use_sigmoid=False,loss_weight=1.6,
            #==============0======1======2=====3=====4====5====6=====7=====8=====9====10====11===12===13====#
            class_weight=[0.999, 1.001, 1.06, 1.07, 1.2, 1.1, 1.02, 1.05, 1.2, 1.05, 1.6, 1.03, 1.1, 1.4])]),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole'))

# pspnet r101 as teacher network, for more detailed usage, please refer to MMSegmentation's docs
t_model_cfgs = dict(
    cfg=[
        # k,  t,  c, s
        [3,   1,  16, 1], # 1/2        0.464K  17.461M
        [3,   4,  32, 2], # 1/4 1      3.44K   64.878M
        [3,   3,  32, 1], #            4.44K   41.772M
        [5,   3,  64, 2], # 1/8 3      6.776K  29.146M
        [5,   3,  64, 1], #            13.16K  30.952M
        [3,   3,  128, 2], # 1/16 5     16.12K  18.369M
        [3,   3,  128, 1], #            41.68K  24.508M
        [5,   6,  160, 2], # 1/32 7     0.129M  36.385M
        [5,   6,  160, 1], #            0.335M  49.298M
        [3,   6,  160, 1], #            0.335M  49.298M
    ],
    channels=[32, 64, 128, 160],
    out_channels=[None, 256, 256, 256],
    embed_out_indice=[2, 4, 6, 9],
    decode_out_indices=[1, 2, 3],
    num_heads=8,
    c2t_stride=2,
)

t_ckpts ='teachers/base_0003_17ce_7-26/iter_75000.pth'
teacher = dict(
    type='mmseg.EncoderDecoder',
    init_cfg=dict(type='Pretrained', checkpoint=t_ckpts),
    backbone=dict(
        type='Topformer_base',
        cfgs=t_model_cfgs['cfg'], 
        channels=t_model_cfgs['channels'],
        out_channels=t_model_cfgs['out_channels'], 
        embed_out_indice=t_model_cfgs['embed_out_indice'],
        decode_out_indices=t_model_cfgs['decode_out_indices'],
        depths=4,
        num_heads=t_model_cfgs['num_heads'],
        c2t_stride=t_model_cfgs['c2t_stride'],
        drop_path_rate=0.1,
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='SimpleHead',
        in_channels=[256, 256, 256],
        in_index=[0, 1, 2],
        channels=256,
        dropout_ratio=0.1,
        num_classes=14,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0),
            dict(type='CrossEntropyLoss', loss_name='loss_entropy',use_sigmoid=False,loss_weight=1.0)]
        ),
    )

# distiller settings
distiller=dict(
    type='SingleTeacherDistiller',                   # distiller name registered
    teacher=teacher,                                 # specify defined teacher to use in the distiller
    teacher_trainable=False,                         # whether to train teacher
    components=[                                     # specify what moudules to calculate kd-loss in teacher and student
        dict(
            # student_module='decode_head.conv_seg',   # student module name
            # teacher_module='decode_head.conv_seg',   # teacher module name
            student_module='decode_head.resizee',   # student module name
            teacher_module='decode_head.resizee',   # teacher module name
            losses=[                                 # specify kd-loss
                dict(
                    type='ChannelWiseDivergence',    # kd-loss type
                    name='loss_cwd_logits',          # name this loss in order to easy get the output of this loss
                    tau=3.0,                           # temperature coefficient
                    loss_weight=1.2,                        # weight of this loss
                )
            ],
            # align_module=''
                        )
    ])


# algorithm settings
algorithm = dict(
    type='GeneralDistill',                                # algorithm name registered
    architecture=dict(                                  # architecture setting
        type='MMSegArchitecture',                       # architecture name registered
        model=student,                                  # specify defined student as the model of architecture
    ),
    #use_gt=True,                                        # whether to calculate gt_loss with gt
    distiller=distiller,                                # specify defined distiller to use in the algorithm
)