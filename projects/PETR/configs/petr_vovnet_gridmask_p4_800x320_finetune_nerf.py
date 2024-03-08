_base_ = './petr_vovnet_gridmask_p4_800x320_finetune.py'
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
ida_aug_conf = {
    'resize_lim': (0.47, 0.625),
    'final_dim': (320, 800),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900,
    'W': 1600,
    'rand_flip': True,
}
backend_args = None
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFilesSwitchRoot',
        to_float32=True,
        backend_args=backend_args,
        data_root_switch=('/proj/adas-data/data/nuscenes',
                          '/proj/agp/renders/real2sim/' +
                          'nusc_train_subset-neurader_no_keyframes_fullres'),
        data_root_switch_p=1.0),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=False,
        training=True),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ])
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
