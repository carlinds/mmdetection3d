#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time 05:00:00
#SBATCH --output /workspaces/%u/logs/mmdet3d/%j.out
#SBATCH --partition zprod
#

CONFIG_FILE=${1:?"No config file given"}
CHECKPOINT_FILE=${2:?"No checkpoint file given"}

singularity exec --nv \
    --bind $PWD:/mmdetection3d \
    --bind /datasets/nuscenes:/datasets/nuscenes \
    --bind nuscenes_custom_files/splits.py:/opt/conda/lib/python3.7/site-packages/nuscenes/utils/splits.py \
    --bind nuscenes_custom_files/loaders.py:/opt/conda/lib/python3.7/site-packages/nuscenes/eval/common/loaders.py \
    --pwd /mmdetection3d \
    ~/workspace/containers/mmdet3d_v1.4.0.sif \
    python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
    ${@:3}

#
#EOF