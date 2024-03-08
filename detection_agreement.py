import argparse
import json
import os
import time
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
    help='Root directory of the NuScenes dataset, e.g. /data/sets/nuscenes')
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

    def __init__(self,
                 nusc: NuScenes,
                 results_a: dict,
                 results_b: dict,
                 eval_version: str = 'detection_cvpr_2019',
                 verbose: bool = True):

        self.nusc = nusc
        self.cfg = config_factory(eval_version)
        self.pred_boxes_a = self.load_prediction(results_a,
                                                 self.cfg.max_boxes_per_sample,
                                                 DetectionBox)
        self.pred_boxes_b = self.load_prediction(results_b,
                                                 self.cfg.max_boxes_per_sample,
                                                 DetectionBox)
        self.verbose = verbose

        assert set(self.pred_boxes_a.sample_tokens) == set(self.pred_boxes_b.sample_tokens), \
            "Samples in pred_boxes_a doesn't match samples in pred_boxes_b."

        # Add center distances.
        self.pred_boxes_a = add_center_dist(nusc, self.pred_boxes_a)
        self.pred_boxes_b = add_center_dist(nusc, self.pred_boxes_b)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes_a = filter_eval_boxes(
            nusc, self.pred_boxes_a, self.cfg.class_range, verbose=verbose)
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
            assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
                'Error: Only <= %d boxes per sample allowed!' % max_boxes_per_sample

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
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.pred_boxes_a, self.pred_boxes_b,
                                class_name, self.cfg.dist_fcn_callable,
                                dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall,
                             self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name,
                                                self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in [
                        'attr_err', 'vel_err', 'orient_err'
                ]:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in [
                        'attr_err', 'vel_err'
                ]:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list


def compute_agreement(results_a, results_b, nusc):
    agreement_metrics = {}
    # Compute detection metrics with a as ground truth and b as predictions.
    detection_eval = DetectionEval(nusc, results_a, results_b)
    a_b_metric_summary, _ = detection_eval.evaluate()
    a_b_metric_summary = a_b_metric_summary.serialize()
    agreement_metrics['a_b_results'] = a_b_metric_summary

    # Compute detection metrics with b as ground truth and a as predictions.
    detection_eval = DetectionEval(nusc, results_b, results_a)
    b_a_metric_summary, _ = detection_eval.evaluate()
    b_a_metric_summary = b_a_metric_summary.serialize()
    agreement_metrics['b_a_results'] = b_a_metric_summary

    # Compute symmetric agreement metrics.
    agreement_metrics['symmetric_map'] = (a_b_metric_summary['mean_ap'] +
                                          b_a_metric_summary['mean_ap']) / 2
    agreement_metrics['symmetric_nds'] = (a_b_metric_summary['nd_score'] +
                                          b_a_metric_summary['nd_score']) / 2
    return agreement_metrics


def main(**kwargs):
    with open(kwargs['results_a'], 'rb') as f_a, open(kwargs['results_b'],
                                                      'rb') as f_b:
        results_a = json.load(f_a)
        results_a = results_a['results']
        results_b = json.load(f_b)
        results_b = results_b['results']

    data_root = kwargs['data_root']
    nusc_version = kwargs['nusc_version']
    output_dir = kwargs['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(data_root), 'Data root does not exist.'
    assert nusc_version in ['v1.0-trainval', 'v1.0-test',
                            'v1.0-mini'], 'Invalid NuScenes version.'
    assert os.path.exists(os.path.join(
        data_root, nusc_version)), 'NuScenes version not found in data root.'

    assert len(results_a) == len(
        results_b), 'Results files do not have the same number of samples.'
    samples = list(results_a.keys())

    nusc = NuScenes(version=nusc_version, dataroot=data_root, verbose=False)
    agreement_results = {}
    for sample in samples:
        agreement_results[sample] = compute_agreement(
            {sample: results_a[sample]}, {sample: results_b[sample]}, nusc)

    with open(os.path.join(output_dir, 'agreement_results.json'), 'w') as f:
        json.dump(agreement_results, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
