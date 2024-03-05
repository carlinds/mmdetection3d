import copy
import json
import os
import pickle
from typing import List

from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric


def create_filtered_ann_file(ann_file: str,
                             filtered_samples: List[str],
                             out_dir: str = None) -> str:
    with open(ann_file, 'rb') as f:
        anns = pickle.load(f)

    new_data_list = [
        sample for sample in anns['data_list']
        if sample['token'] in filtered_samples
    ]
    new_meta_info = copy.deepcopy(anns['metainfo'])

    basename = os.path.basename(ann_file).split('.pkl')[0] + '_filtered.pkl'
    if out_dir:
        new_ann_file = os.path.join(out_dir, basename)
    else:
        os.makedirs('tmp', exist_ok=True)
        new_ann_file = os.path.join('tmp', basename)
    with open(new_ann_file, 'wb') as f:
        pickle.dump({'metainfo': new_meta_info, 'data_list': new_data_list}, f)

    return new_ann_file


data_root = '/proj/adas-data/data/nuscenes'
ann_file = '/proj/adas-data/data/nuscenes/nuscenes_mini_infos_val.pkl'
results_file = (
'outputs/fcos3d_pix2pix_eval_sim_pred_instances/results_nusc.json')
out_dir = 'tmp'
os.makedirs(out_dir, exist_ok=True)
with open(results_file, 'rb') as f:
    results = json.load(f)

filtered_samples = [
    '3e8750f331d7499e9b5123e9eb70f2e2', '3950bd41f74548429c0f7700ff3d8269',
    'c5f58c19249d4137ae063b0e9ecd8b8e'
]
filtered_results = {
    sample: results['results'][sample]
    for sample in filtered_samples
}

filtered_results_file = os.path.join(out_dir, 'filtered_results.json')
with open(filtered_results_file, 'w') as f:
    json.dump({'meta': results['meta'], 'results': filtered_results}, f)
filtered_ann_file = create_filtered_ann_file(ann_file, filtered_samples)

nus_metrics = NuScenesMetric(
    data_root=data_root,
    ann_file=filtered_ann_file,
    metric='bbox',
)

nus_metrics.dataset_meta = {
    'classes': [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ],
    'version':
    'v1.0-trainval'
}

res_dict = {'pred_instances_3d': filtered_results_file}
metrics = nus_metrics.compute_metrics_from_formatted(res_dict)
print(metrics)
