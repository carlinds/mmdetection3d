import argparse
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


def run_compute_agreement(result_a_fp, result_b_fp, output_fp, nusc):
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

    print("Computing agreement...")
    results = []
    agreement_results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for sample in samples:
            res = executor.submit(
                compute_agreement,
                {sample: results_a[sample]},
                {sample: results_b[sample]},
                nusc,
            )
            results.append(res)

        for res in tqdm(results):
            agreement_results.append(res.result())

    agreement_results = {samples[i]: agreement_results[i] for i in range(len(samples))}

    with open(output_fp, "w") as f:
        json.dump(agreement_results, f)

    mean_map = np.mean([res["symmetric_map"] for res in agreement_results.values()])
    mean_nds = np.mean([res["symmetric_nds"] for res in agreement_results.values()])
    print(f"Mean symmetric mAP: {mean_map:.4f}")
    print(f"Mean symmetric NDS: {mean_nds:.4f}")


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

    for result_a_fp, result_b_fp, output_filename in zip(result_a_filepaths, result_b_filepaths, output_filepaths):
        output_fp = os.path.join(output_dir, output_filename)
        run_compute_agreement(result_a_fp, result_b_fp, output_fp, nusc)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
