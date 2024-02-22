#!/bin/bash
singularity shell --nv \
    --bind $PWD:/mmdetection3d \
    --bind /proj:/proj \
    --bind nuscenes_custom_files/splits.py:/opt/conda/lib/python3.7/site-packages/nuscenes/utils/splits.py \
    --bind nuscenes_custom_files/loaders.py:/opt/conda/lib/python3.7/site-packages/nuscenes/eval/common/loaders.py \
    --pwd /mmdetection3d \
    /proj/agp/containers/mmdet3d_v1.4.0.sif