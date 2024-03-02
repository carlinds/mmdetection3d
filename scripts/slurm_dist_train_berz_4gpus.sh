#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --time 01-00:00:00
#SBATCH --output /proj/agp/users/%u/logs/%j.out
#SBATCH -A Berzelius-2024-51
#

NAME=${1:?"No name given"}
CONFIG_FILE=${2:?"No config file given"}
DATA_ROOT_TRAIN=${3:?"No train data root given"}
DATA_ROOT_VAL=${4:?"No val data root given"}
wandb_group=${WANDB_GROUP:-""}

OUTPUT_DIR=outputs/train/${wandb_group}/${NAME}

export PORT=$RANDOM
export WANDB_NAME=${NAME}
export WANDB_RUN_GROUP=${wandb_group}

train_pickle=nuscenes_infos_train.pkl
eval_pickle=nuscenes_infos_val_clear.pkl

singularity exec --nv \
    --bind $PWD:/mmdetection3d \
    --bind /proj:/proj \
    --bind nuscenes_custom_files/splits.py:/opt/conda/lib/python3.7/site-packages/nuscenes/utils/splits.py \
    --bind nuscenes_custom_files/loaders.py:/opt/conda/lib/python3.7/site-packages/nuscenes/eval/common/loaders.py \
    --pwd /mmdetection3d \
    /proj/agp/containers/mmdet3d_v1.4.0_220224.sif \
    bash ./tools/dist_train.sh \
        ${CONFIG_FILE} \
        4 \
        --work-dir ${OUTPUT_DIR} \
        --cfg-options val_evaluator.jsonfile_prefix=${OUTPUT_DIR} test_evaluator.jsonfile_prefix=${OUTPUT_DIR} \
        train_dataloader.dataset.data_root=${DATA_ROOT_TRAIN} \
        train_dataloader.dataset.ann_file=${train_pickle} \
        val_dataloader.dataset.data_root=${DATA_ROOT_VAL} \
        val_dataloader.dataset.ann_file=${eval_pickle} \
        test_dataloader.dataset.data_root=${DATA_ROOT_VAL} \
        test_dataloader.dataset.ann_file=${eval_pickle} \
        val_evaluator.data_root=${DATA_ROOT_VAL} \
        val_evaluator.ann_file=${DATA_ROOT_VAL}/${eval_pickle} \
        test_evaluator.data_root=${DATA_ROOT_VAL} \
        test_evaluator.ann_file=${DATA_ROOT_VAL}/${eval_pickle} \
        ${@:5}

#
#EOF