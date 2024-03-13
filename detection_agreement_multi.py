import argparse
from collections import defaultdict
import json
import os

from detection_agreement import DetectionEval, compute_agreement

import numpy as np
from nuscenes import NuScenes
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(
    description="Compute detection agreement between two sets of predictions"
)
parser.add_argument(
    "--results_a", type=str, help="Path to the txt file with first results file"
)
parser.add_argument(
    "--results_b", type=str, help="Path to the txt file with second results file"
)
parser.add_argument(
    "--data_root",
    type=str,
    required=True,
    help="Root directory of the NuScenes dataset, e.g. /data/sets/nuscenes",
)
parser.add_argument(
    "--nusc_version", type=str, default="v1.0-trainval", help="NuScenes version to use"
)
parser.add_argument(
    "--output_dir", type=str, default="tmp", help="Output directory for the results"
)
parser.add_argument(
    "--output_file",
    type=str,
    help="Path to txt file specifying the output file for each sample. ",
)
parser.add_argument(
    "--max_workers",
    type=int,
    default=16,
    help="Number of workers to use for parallel processing",
)
parser.add_argument(
    "--aggregate_per_scene", action="store_true", help="Aggregate results per scene"
)


def run_compute_agreement(
    result_a_fp, result_b_fp, output_fp, nusc, aggregate_per_scene
):
    with open(result_a_fp, "rb") as f_a, open(result_b_fp, "rb") as f_b:
        results_a = json.load(f_a)
        results_a = results_a["results"]
        print("Loaded results A from {}".format(result_a_fp))

        results_b = json.load(f_b)
        results_b = results_b["results"]
        print("Loaded results B from {}".format(result_b_fp))

    assert len(results_a) == len(
        results_b
    ), "Results files do not have the same number of samples."
    samples = list(results_a.keys())
    agreement_results = {}

    scene_to_samples = defaultdict(dict)
    for sample in samples:
        scene_token = (
            nusc.get("sample", sample)["scene_token"] if aggregate_per_scene else sample
        )
        if scene_token not in scene_to_samples:
            scene_to_samples[scene_token]["results_a"] = {}
            scene_to_samples[scene_token]["results_b"] = {}

        scene_to_samples[scene_token]["results_a"][sample] = results_a[sample]
        scene_to_samples[scene_token]["results_b"][sample] = results_b[sample]

    scene_tokens = list(scene_to_samples.keys())

    print("Computing agreement...")
    futures = []

    agreement_results = dict()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for scene_token in scene_tokens:
            futures.append(
                executor.submit(
                    compute_agreement,
                    scene_to_samples[scene_token]["results_a"],
                    scene_to_samples[scene_token]["results_b"],
                    nusc,
                )
            )

        for i, res in enumerate(tqdm(futures)):
            agreement_results[scene_tokens[i]] = res.result()

    if aggregate_per_scene:
        output_fp = output_fp.replace(".json", "_per_scene.json")
    with open(output_fp, "w") as f:
        json.dump(agreement_results, f)
    print("Saved results to {}".format(output_fp))


def main(**kwargs):
    print("Opening results files...")
    with open(kwargs["results_a"], "rb") as f_a, open(
        kwargs["results_b"], "rb"
    ) as f_b, open(kwargs["output_file"], "r") as f_out:
        result_a_filepaths = [x.strip() for x in f_a.readlines()]
        result_b_filepaths = [x.strip() for x in f_b.readlines()]
        output_filepaths = [x.strip() for x in f_out.readlines()]

    data_root = kwargs["data_root"]
    nusc_version = kwargs["nusc_version"]
    output_dir = kwargs["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(data_root), "Data root does not exist."
    assert nusc_version in [
        "v1.0-trainval",
        "v1.0-test",
        "v1.0-mini",
    ], "Invalid NuScenes version."
    assert os.path.exists(
        os.path.join(data_root, nusc_version)
    ), "NuScenes version not found in data root."

    print("Loading NuScenes...")
    nusc = NuScenes(version=nusc_version, dataroot=data_root, verbose=False)

    for result_a_fp, result_b_fp, output_filename in zip(
        result_a_filepaths, result_b_filepaths, output_filepaths
    ):
        output_fp = os.path.join(output_dir, output_filename)
        run_compute_agreement(
            result_a_fp, result_b_fp, output_fp, nusc, kwargs["aggregate_per_scene"]
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
