#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 02:00:00
#SBATCH --output /proj/agp/users/%u/logs/%j.out
#SBATCH -A Berzelius-2024-45
#

export WANDB_ENTITY=agp

singularity exec --nv \
  --bind $PWD:/nerfstudio \
  --bind /proj:/proj \
  --pwd /nerfstudio \
  /proj/agp/containers/mmdet3d_v1.4.0_220224.sif \
  $@

#
#EOF
