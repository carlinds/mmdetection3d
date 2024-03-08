_base_ = './petr_vovnet_gridmask_p4_800x320.py'
num_epochs = 12
train_cfg = dict(max_epochs=num_epochs, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=2e-4))
load_from = 'pretrained/petr_vovnet_gridmask_p4_800x320-e2191752.pth'
default_hooks = dict(checkpoint=dict(interval=1))
