#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 00-01:00:00
#SBATCH --output /proj/agp/users/%u/logs/%j.out
#SBATCH -A Berzelius-2023-209
#

NAME=${1:?"No name given"}
CONFIG_FILE=${2:?"No config file given"}

OUTPUT_DIR=outputs/train/${NAME}

singularity exec --nv \
    --bind $PWD:/mmdetection3d \
    --bind /proj/adas-data/data/nuscenes:/data/nuscenes \
    --pwd /mmdetection3d \
    ~/workspace/containers/mmdet3d_v1.4.0.sif \
    python -u tools/train.py \
        ${CONFIG_FILE} \
        --work-dir ${OUTPUT_DIR} \
        --cfg-options val_evaluator.jsonfile_prefix=${OUTPUT_DIR} test_evaluator.jsonfile_prefix=${OUTPUT_DIR} \
        ${@:4}

#
#EOF