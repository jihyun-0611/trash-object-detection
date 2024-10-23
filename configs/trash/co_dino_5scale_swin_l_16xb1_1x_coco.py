_base_ = '/data/ephemeral/home/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_1x_coco.py'

num_classes = 10

data_root = '/data/ephemeral/home/dataset/'

metainfo = {
    'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
}

load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='RandomResize',
        scale=_base_.image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=_base_.image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=_base_.image_size, pad_val=dict(img=(114, 114, 114))),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        dataset=dict(
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train_split_random.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=load_pipeline, 
            backend_args=_base_.backend_args,
            )))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=_base_.image_size, keep_ratio=True),
    dict(type='Pad', size=_base_.image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val_split_random.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'val_split_random.json')
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=8,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
    )

load_from='https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth'

work_dir = './work_dirs/co_dino_5scale_swin_l_lsj_16xb1_1x_coco'

# wandb init arguments : 필요 없으면 모두 None으로 해도됨 
run_name = 'co_dino_5scale_swin_l_lsj_16xb1_1x_coco'
tags = ['co-dter', 'swin', 'lsj', '12epoch'] # 원하는 태그 설정
notes = 'training co-detr using swin backbone with data augementations(lsj)' # 해당 run에 대한 설명

# visualizer 설정 
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={'project': 'MMDetection-OD', 
                            'entity': 'buan99-personal', 
                            'name': run_name, 
                            'tags': tags, 
                            'notes': notes},
         )
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=True, interval=1, score_thr=0.3), 
)


