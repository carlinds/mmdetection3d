WANDB_GROUP=test_shifted_fcos3d sbatch scripts/single_test_berzelius.sh fcos3d_pix2pix_shifted_-2_eval_sim configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune_further_pix2pix.py outputs/train/sweeps_fcos3d_w_pix2pix_p1_eval_render/optim_wrapper.optimizer.lr_1e-4_resumed/epoch_6.pth /proj/agp/renders/real2sim/nuscenes_shifted_-2 nuscenes_infos_val_clear_shifted.pkl val_evaluator.nusc2nerf_transform_path=aggregated_dataparser_transforms.json test_evaluator.nusc2nerf_transform_path=aggregated_dataparser_transforms.json val_evaluator.shift="[-2.0,0.0,0.0]" test_evaluator.shift="[-2,0.0,0.0]"
WANDB_GROUP=test_shifted_fcos3d sbatch scripts/single_test_berzelius.sh fcos3d_pix2pix_shifted_-1_eval_sim configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune_further_pix2pix.py outputs/train/sweeps_fcos3d_w_pix2pix_p1_eval_render/optim_wrapper.optimizer.lr_1e-4_resumed/epoch_6.pth /proj/agp/renders/real2sim/nuscenes_shifted_-1 nuscenes_infos_val_clear_shifted.pkl val_evaluator.nusc2nerf_transform_path=aggregated_dataparser_transforms.json test_evaluator.nusc2nerf_transform_path=aggregated_dataparser_transforms.json val_evaluator.shift="[-1.0,0.0,0.0]" test_evaluator.shift="[-1,0.0,0.0]"
WANDB_GROUP=test_shifted_fcos3d sbatch scripts/single_test_berzelius.sh fcos3d_pix2pix_shifted_1_eval_sim configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune_further_pix2pix.py outputs/train/sweeps_fcos3d_w_pix2pix_p1_eval_render/optim_wrapper.optimizer.lr_1e-4_resumed/epoch_6.pth /proj/agp/renders/real2sim/nuscenes_shifted_1 nuscenes_infos_val_clear_shifted.pkl val_evaluator.nusc2nerf_transform_path=aggregated_dataparser_transforms.json test_evaluator.nusc2nerf_transform_path=aggregated_dataparser_transforms.json val_evaluator.shift="[1.0,0.0,0.0]" test_evaluator.shift="[1,0.0,0.0]"
WANDB_GROUP=test_shifted_fcos3d sbatch scripts/single_test_berzelius.sh fcos3d_pix2pix_shifted_2_eval_sim configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune_further_pix2pix.py outputs/train/sweeps_fcos3d_w_pix2pix_p1_eval_render/optim_wrapper.optimizer.lr_1e-4_resumed/epoch_6.pth /proj/agp/renders/real2sim/nuscenes_shifted_2 nuscenes_infos_val_clear_shifted.pkl val_evaluator.nusc2nerf_transform_path=aggregated_dataparser_transforms.json test_evaluator.nusc2nerf_transform_path=aggregated_dataparser_transforms.json val_evaluator.shift="[2.0,0.0,0.0]" test_evaluator.shift="[2,0.0,0.0]"
WANDB_GROUP=test_shifted_fcos3d sbatch scripts/single_test_berzelius.sh fcos3d_pix2pix_shifted_0_eval_sim configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune_further_pix2pix.py outputs/train/sweeps_fcos3d_w_pix2pix_p1_eval_render/optim_wrapper.optimizer.lr_1e-4_resumed/epoch_6.pth /proj/agp/renders/real2sim/nusc_val_subset-neurader_no_keyframes_fullres nuscenes_infos_val_clear_shifted.pkl val_evaluator.nusc2nerf_transform_path=aggregated_dataparser_transforms.json test_evaluator.nusc2nerf_transform_path=aggregated_dataparser_transforms.json val_evaluator.shift="[0.0,0.0,0.0]" test_evaluator.shift="[0,0.0,0.0]"
