#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --cpus-per-task 64
#SBATCH --time 02-00:00:00
#SBATCH --output /workspaces/%u/logs/mmdet3d/%j.out
#SBATCH --partition zprod
#

NAME=${1:?"No name given"}
CONFIG_FILE=${2:?"No config file given"}

OUTPUT_DIR=outputs/train/${NAME}

export PORT=$RANDOM
export WANDB_NAME=${NAME}

singularity exec --nv \
    --bind $PWD:/mmdetection3d \
    --bind /datasets/nuscenes:/datasets/nuscenes \
    --pwd /mmdetection3d \
    ~/workspace/containers/mmdet3d_v1.4.0.sif \
    bash ./tools/dist_train.sh \
        ${CONFIG_FILE} \
        8 \
        --work-dir ${OUTPUT_DIR} \
        ${@:3}

#
#EOF