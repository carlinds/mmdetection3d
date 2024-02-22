import argparse
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test metrics from pre-computed results')
    parser.add_argument(
        '--result_file', type=str, help='The result file in json format')
    parser.add_argument(
        '--data_root', type=str, help='The root of the data')
    parser.add_argument(
        '--ann_file', type=str, help='The annotation file')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    nus_metrics = NuScenesMetric(
        data_root=args.data_root,
        ann_file=args.ann_file,
        metric='bbox',
    )
    nus_metrics.dataset_meta = {
        "classes": [
            "car",
            "truck",
            "trailer",
            "bus",
            "construction_vehicle",
            "bicycle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "barrier"
        ],
        "version": "v1.0-trainval"
    }


    res_dict = {"pred_instances_3d": args.result_file}

    metrics = nus_metrics.compute_metrics_from_formatted(res_dict)
    print(metrics)


if __name__ == '__main__':
    main()