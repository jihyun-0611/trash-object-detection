_base_ = '/data/ephemeral/home/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py'


num_classes = 10

data_root = '/data/ephemeral/home/dataset/'

metainfo = {
    'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
}

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_split_random.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=_base_.train_pipeline,
        backend_args=_base_.backend_args))

val_dataloader = dict(
    dataset=dict(
        pipeline=_base_.test_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val_split_random.json',
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

# test_dataloader = dict(
#     dataset=dict(
#         pipeline=_base_.test_pipeline,
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='test.json',
#         data_prefix=dict(img=''),
#         test_mode=True,
#         ))

val_evaluator = dict(ann_file=data_root + 'val_split_random.json')
test_evaluator = val_evaluator
# test_evaluator = dict(
#     type='CocoMetric',
#     format_only=True,
#     ann_file=data_root + 'test.json',
#     outfile_prefix='/data/ephemeral/home/mmdetection/results/co-detr-results'
# )

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=4,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))


load_from='https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_r50_1x_coco-7481f903.pth'

work_dir = './work_dirs/co_dino_5scale_r50_8xb2_1x_coco'

# wandb init arguments : 필요 없으면 모두 None으로 해도됨 
run_name = 'co_dino_5scale_r50_8xb2_1x_coco'
tags = ['co-dter', 'resnet50', '12epoch'] # 원하는 태그 설정
notes = 'visualization feature map in co-dter' # 해당 run에 대한 설명

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
)

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
