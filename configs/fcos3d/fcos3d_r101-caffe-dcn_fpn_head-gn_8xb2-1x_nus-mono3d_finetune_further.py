_base_ = './fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py'
num_epochs = 6
train_cfg = dict(max_epochs=num_epochs, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=2e-5))
load_from = 'pretrained/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'
env_cfg = dict(
    dist_cfg=dict(backend='nccl',timeout=3600),
)