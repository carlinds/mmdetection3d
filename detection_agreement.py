import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import (DetectionBox,
                                                  DetectionMetricDataList,
                                                  DetectionMetrics)
from nuscenes.eval.detection.render import visualize_sample
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Compute detection agreement between two sets of predictions')
parser.add_argument(
    '--results_a', type=str, help='Path to the first results file')
parser.add_argument(
    '--results_b', type=str, help='Path to the second results file')
parser.add_argument(
    '--data_root',
    type=str,
    required=True,
    help='Root directory of the NuScenes dataset, e.g. /data/sets/nuscenes',
)
parser.add_argument(
    '--nusc_version',
    type=str,
    default='v1.0-trainval',
    help='NuScenes version to use')
parser.add_argument(
    '--output_dir',
    type=str,
    default='tmp',
    help='Output directory for the results')
parser.add_argument(
    '--output_file',
    type=str,
    default='agreement_results.json',
    help='Name of the output file',
)
parser.add_argument(
    '--max_workers',
    type=int,
    default=16,
    help='Number of workers to use for parallel processing',
)
parser.add_argument(
    '--aggregate_per_scene',
    action='store_true',
    help='Aggregate results per scene')
parser.add_argument(
    '--aggregate_all', action='store_true', help='Aggregate results per scene')
parser.add_argument(
    '--nusc2nerf_transformations',
    type=str,
    default='',
    help='Path to the file with the nuscenes to nerf transformations',
)
parser.add_argument(
    '--range_fraction',
    default=1.0,
    type=float,
)
parser.add_argument(
    '--dist_ths',
    nargs='+',
    default=[0.5, 1.0, 2.0, 4.0],
    type=float,
)
parser.add_argument(
    '--confidence_ths',
    default=0.5,
    type=float,
)


class DetectionEval:
    """This is the official nuScenes detection evaluation code. Results are
    written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """

    def __init__(
        self,
        nusc: NuScenes,
        results_a: dict,
        results_b: dict,
        eval_version: str = 'detection_cvpr_2019',
        range_fraction: float = 1.0,
        dist_ths: list = [0.5, 1.0, 2.0, 4.0],
        verbose: bool = True,
    ):

        self.nusc = nusc
        self.cfg = config_factory(eval_version)
        self.cfg.class_range = {
            class_name: (class_range * range_fraction)
            for class_name, class_range in self.cfg.class_range.items()
        }
        self.cfg.dist_ths = dist_ths
        self.pred_boxes_a = self.load_prediction(results_a,
                                                 self.cfg.max_boxes_per_sample,
                                                 DetectionBox)
        self.pred_boxes_b = self.load_prediction(results_b,
                                                 self.cfg.max_boxes_per_sample,
                                                 DetectionBox)
        self.verbose = verbose

        assert set(self.pred_boxes_a.sample_tokens) == set(
            self.pred_boxes_b.sample_tokens
        ), "Samples in pred_boxes_a doesn't match samples in pred_boxes_b."

        # Add center distances.
        self.pred_boxes_a = add_center_dist(nusc, self.pred_boxes_a)
        self.pred_boxes_b = add_center_dist(nusc, self.pred_boxes_b)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        if len(self.pred_boxes_a.all):
            self.pred_boxes_a = filter_eval_boxes(
                nusc, self.pred_boxes_a, self.cfg.class_range, verbose=verbose)
        if len(self.pred_boxes_b.all):
            self.pred_boxes_b = filter_eval_boxes(
                nusc, self.pred_boxes_b, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.pred_boxes_a.sample_tokens

    def load_prediction(self, results: dict, max_boxes_per_sample: int,
                        box_cls) -> EvalBoxes:
        """Loads object predictions from dict.

        :param results: Dict of results.
        :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
        :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
        :return: The deserialized results and meta data.
        """

        # Deserialize results and get meta data.
        all_results = EvalBoxes.deserialize(results, box_cls)

        # Check that each sample has no more than x predicted boxes.
        for sample_token in all_results.sample_tokens:
            assert len(
                all_results.boxes[sample_token]) <= max_boxes_per_sample, (
                    'Error: Only <= %d boxes per sample allowed!' %
                    max_boxes_per_sample)

        return all_results

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """Performs the actual evaluation.

        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        valids = {}
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(
                    self.pred_boxes_a,
                    self.pred_boxes_b,
                    class_name,
                    self.cfg.dist_fcn_callable,
                    dist_th,
                )
                metric_data_list.set(class_name, dist_th, md)
                valids[(class_name, dist_th)] = (
                    len([
                        1 for gt_box in self.pred_boxes_a.all
                        if gt_box.detection_name == class_name
                    ]) > 0)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                if not valids[(class_name, dist_th)]:
                    continue
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall,
                             self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                if not valids[(class_name, self.cfg.dist_th_tp)]:
                    continue
                metric_data = metric_data_list[(class_name,
                                                self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in [
                        'attr_err',
                        'vel_err',
                        'orient_err',
                ]:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in [
                        'attr_err',
                        'vel_err',
                ]:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list


def compute_agreement(
    results_a,
    results_b,
    nusc,
    shifts=defaultdict(lambda: (0.0, 0.0, 0.0)),
    range_fraction=1.0,
    dist_ths=[0.5, 1.0, 2.0, 4.0],
    results_a_conf_threshold=0.5,
    results_b_conf_threshold=0.5,
    verbose=False,
    vis=False,
):
    agreement_metrics = {}
    # Compute detection metrics with a as ground truth and b as predictions.
    thresholded_results_a = {}
    for sample_token, boxes in results_a.items():
        thresholded_results_a[sample_token] = [
            box for box in boxes
            if box['detection_score'] > results_a_conf_threshold
        ]

    thresholded_results_b = {}
    for sample_token, boxes in results_b.items():
        for box in boxes:
            box['translation'] = [
                t + s for t, s in zip(box['translation'], shifts[sample_token])
            ]
        results_b[sample_token] = boxes
        thresholded_boxes = [
            box for box in boxes
            if box['detection_score'] > results_b_conf_threshold
        ]
        thresholded_results_b[sample_token] = thresholded_boxes

    detection_eval = DetectionEval(
        nusc,
        thresholded_results_a,
        results_b,
        range_fraction=range_fraction,
        dist_ths=dist_ths,
        verbose=verbose)
    a_b_metric_summary, _ = detection_eval.evaluate()
    a_b_metric_summary = a_b_metric_summary.serialize()
    agreement_metrics['a_b_results'] = a_b_metric_summary
    if vis:
        visualize_sample(
            detection_eval.nusc,
            detection_eval.sample_tokens[0],
            detection_eval.pred_boxes_a,
            detection_eval.pred_boxes_b,
            savepath=f'vis/{detection_eval.sample_tokens[0]}_real_vs_sim.png',
        )

    # Compute detection metrics with b as ground truth and a as predictions.
    detection_eval = DetectionEval(
        nusc,
        thresholded_results_b,
        results_a,
        range_fraction=range_fraction,
        dist_ths=dist_ths,
        verbose=verbose)
    b_a_metric_summary, _ = detection_eval.evaluate()
    b_a_metric_summary = b_a_metric_summary.serialize()
    agreement_metrics['b_a_results'] = b_a_metric_summary
    if vis:
        visualize_sample(
            detection_eval.nusc,
            detection_eval.sample_tokens[0],
            detection_eval.pred_boxes_a,
            detection_eval.pred_boxes_b,
            savepath=f'vis/{detection_eval.sample_tokens[0]}_sim_vs_real.png',
        )

    # Compute symmetric agreement metrics.
    agreement_metrics['symmetric_map'] = (a_b_metric_summary['mean_ap'] +
                                          b_a_metric_summary['mean_ap']) / 2
    agreement_metrics['symmetric_nds'] = (a_b_metric_summary['nd_score'] +
                                          b_a_metric_summary['nd_score']) / 2
    return agreement_metrics


def run_compute_agreement(result_a_fp, result_b_fp, output_fp, nusc,
                          aggregate_per_scene, aggregate_all, range_fraction,
                          dist_ths, confidence_ths, nusc2nerf_transformations):
    print('Opening results files...')
    with open(result_a_fp, 'rb') as f_a, open(result_b_fp, 'rb') as f_b:
        results_a = json.load(f_a)
        results_a = results_a['results']
        print(f'Loaded results A from {result_a_fp}')

        results_b = json.load(f_b)
        results_b = results_b['results']
        print(f'Loaded results B from {result_b_fp}')

        if 'shifted' in result_b_fp:
            shift = float(
                result_b_fp.rstrip('/pred_instances_3d/results_nusc.json').
                split('shifted_')[-1])
            print('Found shift in result name. Shift: ', shift)
        else:
            shift = 0.0

    assert len(results_a) == len(
        results_b), 'Results files do not have the same number of samples.'
    samples = list(results_a.keys())
    scene_to_samples = defaultdict(dict)
    for sample in samples:
        scene_name = nusc.get('scene',
                              nusc.get('sample',
                                       sample)['scene_token'])['name']
        nusc2nerf = np.array(
            nusc2nerf_transformations[scene_name.split('-')[1]])
        shift_in_nerf = np.array([shift, 0.0, 0.0])
        shift_in_nuscenes = nusc2nerf[:3, :3].T @ shift_in_nerf.reshape(-1, 1)
        scene_token = (
            nusc.get('sample', sample)['scene_token']
            if aggregate_per_scene else sample)
        if scene_token not in scene_to_samples:
            scene_to_samples[scene_token]['results_a'] = {}
            scene_to_samples[scene_token]['results_b'] = {}
            scene_to_samples[scene_token]['shifts'] = {}

        scene_to_samples[scene_token]['results_a'][sample] = results_a[sample]
        scene_to_samples[scene_token]['results_b'][sample] = results_b[sample]
        scene_to_samples[scene_token]['shifts'][sample] = tuple(
            shift_in_nuscenes.flatten().tolist())

    scene_tokens = list(scene_to_samples.keys())

    print('Computing agreement...')
    futures = []
    agreement_results = dict()

    if aggregate_all:
        results_a = {
            token: scene_to_samples[token]['results_a'][token]
            for token in scene_tokens
        }
        results_b = {
            token: scene_to_samples[token]['results_b'][token]
            for token in scene_tokens
        }
        shifts = {
            token: scene_to_samples[token]['shifts'][token]
            for token in scene_tokens
        }

        agreement_results = compute_agreement(
            results_a,
            results_b,
            nusc,
            shifts,
            range_fraction=range_fraction,
            dist_ths=dist_ths,
            results_a_conf_threshold=confidence_ths,
            results_b_conf_threshold=confidence_ths)

    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for scene_token in scene_tokens:
                futures.append(
                    executor.submit(
                        compute_agreement,
                        scene_to_samples[scene_token]['results_a'],
                        scene_to_samples[scene_token]['results_b'],
                        nusc,
                        scene_to_samples[scene_token]['shifts'],
                        range_fraction,
                        dist_ths,
                        confidence_ths,
                        confidence_ths,
                    ))

            for i, res in enumerate(tqdm(futures)):
                agreement_results[scene_tokens[i]] = res.result()

    if aggregate_per_scene:
        output_fp = output_fp.replace('.json', '_per_scene.json')

    if aggregate_all:
        output_fp = output_fp.replace('.json', '_all.json')

    if range_fraction:
        output_fp = output_fp.replace(
            '.json', f'_range_fraction_{range_fraction}.json')

    if dist_ths:
        ths_str = '_'.join([str(x) for x in dist_ths])
        output_fp = output_fp.replace('.json', f'_dist_ths_{ths_str}.json')

    if confidence_ths:
        output_fp = output_fp.replace('.json',
                                      f'_conf_ths_{confidence_ths}.json')

    with open(output_fp, 'w') as f:
        json.dump(agreement_results, f)
    print('Saved results to {}'.format(output_fp))


def main(**kwargs):
    assert not (
        kwargs['aggregate_per_scene'] and kwargs['aggregate_all']
    ), 'Cannot aggregate per scene and all samples at the same time.'

    if kwargs['results_a'].endswith('.json'):
        print('Running detection agreement for a single file.')
        assert kwargs['results_b'].endswith(
            '.json'), 'Results B must also be a JSON file.'
        assert kwargs['output_file'].endswith(
            '.json'), 'Output file must be a JSON file.'
        result_a_filepaths = [kwargs['results_a']]
        result_b_filepaths = [kwargs['results_b']]
        output_filepaths = [kwargs['output_file']]

    elif kwargs['results_a'].endswith('.txt'):
        print('Running detection agreement for a list of files.')
        assert kwargs['results_b'].endswith(
            '.txt'
        ), 'Results B must also be a .txt file with a list of result files.'
        assert kwargs['output_file'].endswith(
            '.txt'
        ), 'Output file must be a .txt file with a list of output filenames.'
        with open(kwargs['results_a'],
                  'r') as f_a, open(kwargs['results_b'],
                                    'r') as f_b, open(kwargs['output_file'],
                                                      'r') as f_out:
            result_a_filepaths = [x.strip() for x in f_a.readlines()]
            result_b_filepaths = [x.strip() for x in f_b.readlines()]
            output_filepaths = [x.strip() for x in f_out.readlines()]

    data_root = kwargs['data_root']
    nusc_version = kwargs['nusc_version']
    output_dir = kwargs['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(data_root), 'Data root does not exist.'
    assert nusc_version in [
        'v1.0-trainval',
        'v1.0-test',
        'v1.0-mini',
    ], 'Invalid NuScenes version.'
    assert os.path.exists(os.path.join(
        data_root, nusc_version)), 'NuScenes version not found in data root.'

    print('Loading NuScenes...')
    nusc = NuScenes(version=nusc_version, dataroot=data_root, verbose=False)
    print('Loaded NuScenes.')

    if len(kwargs['nusc2nerf_transformations']):
        print('Loading transformations...')
        with open(kwargs['nusc2nerf_transformations'], 'r') as f:
            nusc2nerf_transformations = json.load(f)
    else:
        nusc2nerf_transformations = defaultdict(lambda: np.eye(4))

    for result_a_fp, result_b_fp, output_filename in zip(
            result_a_filepaths, result_b_filepaths, output_filepaths):
        output_fp = os.path.join(output_dir, output_filename)
        run_compute_agreement(result_a_fp, result_b_fp, output_fp, nusc,
                              kwargs['aggregate_per_scene'],
                              kwargs['aggregate_all'],
                              kwargs['range_fraction'], kwargs['dist_ths'],
                              kwargs['confidence_ths'],
                              nusc2nerf_transformations)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
