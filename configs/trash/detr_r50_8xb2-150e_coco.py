_base_ = '../detr/detr_r50_8xb2-150e_coco.py'

class_name = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
              "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
num_classes = len(class_name)

data_root = '/data/ephemeral/home/dataset/'

model = dict(
    bbox_head=dict(num_classes=10)
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(480, 480), (512, 512), (544, 544), (576, 576),
                        (608, 608), (640, 640), (672, 672), (704, 704),
                        (736, 736), (768, 768), (800, 800)],
                keep_ratio=True)
        ],
        [
            dict(
                type='RandomChoiceResize',
                scales=[(400, 400), (500, 500), (600, 600)],
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(384, 384),
                allow_negative_crop=True),
            dict(
                type='RandomChoiceResize',
                scales=[(480, 480), (512, 512), (544, 544),
                        (576, 576), (608, 608), (640, 640),
                        (672, 672), (704, 704), (736, 736),
                        (768, 768), (800, 800)],
                keep_ratio=True)
        ]]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]


train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_root=data_root,
        ann_file='train_split_random.json',
        data_prefix=dict(img=''),
        metainfo=dict(classes=class_name),
        pipeline=train_pipeline,
        ))
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        ann_file='val_split_random.json',
        data_prefix=dict(img=''),
        metainfo=dict(classes=class_name),
        pipeline=test_pipeline,
        ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_split_random.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator

fp16 = dict(loss_scale=512.0)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

# learning policy
max_epochs = 50

work_dir = './work_dirs/detr_r50_8xb2-150e_coco'

# wandb init arguments : 필요 없으면 모두 None으로 해도됨 
run_name = 'detr_r50_8xb2-150e_coco'
tags = ['dter', 'resnet50', '50epoch'] # 원하는 태그 설정
notes = 'training detr to compare with co-detr' # 해당 run에 대한 설명

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
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
