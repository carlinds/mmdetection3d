_base_ = './fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py'
# model settings
model = dict(
    train_cfg=dict(
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05]))
# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.001))
load_from = 'pretrained/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210715_235813-4bed5239.pth'
default_hooks = dict(checkpoint=dict(interval=1))
