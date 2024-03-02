#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 03:00:00
#SBATCH --output /proj/agp/users/%u/logs/%j.out
#SBATCH -A Berzelius-2023-209
#

NAME=${1:?"No name given"}
CONFIG_FILE=${2:?"No config file given"}
CHECKPOINT_FILE=${3:?"No checkpoint file given"}
DATA_ROOT=${4:?"No data root given"}
ANN_FILE=${5:?"No annotation file given"}

OUTPUT_DIR=outputs/$(basename $DATA_ROOT)_${ANN_FILE%.pkl}_${NAME}

singularity exec --nv \
    --bind $PWD:/mmdetection3d \
    --bind $DATA_ROOT:/mmdetection3d/data/nuscenes \
    --bind nuscenes_custom_files/splits.py:/opt/conda/lib/python3.7/site-packages/nuscenes/utils/splits.py \
    --bind nuscenes_custom_files/loaders.py:/opt/conda/lib/python3.7/site-packages/nuscenes/eval/common/loaders.py \
    --pwd /mmdetection3d \
    /proj/agp/containers/mmdet3d_v1.4.0_220224.sif \
    python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
    --ann_file ${ANN_FILE} \
    --work-dir ${OUTPUT_DIR} \
    --cfg-options val_evaluator.jsonfile_prefix=${OUTPUT_DIR} test_evaluator.jsonfile_prefix=${OUTPUT_DIR}
    ${@:3}

#
#EOF