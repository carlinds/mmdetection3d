_base_ = './fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune_further.py'
backend_args = None
train_pipeline = [
    dict(
        type='LoadImageFromFileMono3DSwitchRoot',
        backend_args=backend_args,
        data_root_switch=("/proj/adas-data/data/nuscenes","/proj/agp/renders/real2sim/nusc_pix2pixhd"),
        data_root_switch_p=1.0),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='mmdet.Resize', scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
