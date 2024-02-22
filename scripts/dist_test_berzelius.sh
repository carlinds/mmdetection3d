#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --time 03:00:00
#SBATCH --output /proj/agp/users/%u/logs/%j.out
#SBATCH -A Berzelius-2023-211
#

CONFIG_FILE=${1:?"No config file given"}
CHECKPOINT_FILE=${2:?"No checkpoint file given"}

singularity exec --nv \
    --bind $PWD:/mmdetection3d \
    --bind /proj:/proj \
    /proj/agp/containers/mmdet3d_v1.4.0.sif \
    ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} 4\
    ${@:3}

#
#EOF