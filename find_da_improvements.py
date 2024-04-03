import json
import math

import numpy as np

without_aug_file = 'outputs/detection_agreement/bevformer/per_sample/bevformer_detection_agreement_ft_thres05_range_fraction_1.0_dist_ths_2.0.json'
with_aug_file = 'outputs/detection_agreement/bevformer/per_sample/bevformer_detection_agreement_ft_pix2pix_thres05_range_fraction_1.0_dist_ths_2.0.json'

with open(without_aug_file, 'r') as f:
    without_aug = json.load(f)
with open(with_aug_file, 'r') as f:
    with_aug = json.load(f)

sample_tokens = list(without_aug.keys())
assert sample_tokens == list(with_aug.keys())

da_diff = []
for sample_token in sample_tokens:
    da_without_aug = without_aug[sample_token]['symmetric_nds']
    da_with_aug = with_aug[sample_token]['symmetric_nds']

    if math.isnan(da_with_aug):
        da_with_aug = 0.0
    if math.isnan(da_without_aug):
        da_without_aug = 0.0

    da_diff.append(da_with_aug - da_without_aug)

da_diff = np.array(da_diff)
sorted_indices = np.argsort(da_diff)[::-1]
sorted_sample_tokens = np.array(sample_tokens)[sorted_indices]

# Save to .txt file
with open('sample_tokens_sorted_by_da_improvements.txt', 'w') as f:
    for sample_token in sorted_sample_tokens:
        f.write(sample_token + '\n')
