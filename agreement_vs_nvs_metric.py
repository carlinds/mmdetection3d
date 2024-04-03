import json
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import nuscenes
import pandas as pd
import seaborn as sns

MODELS = ['bevformer']
AGGREGATION_METHOD = 'per_scene'
AUGS = ['', '_aug', '_nerf', '_pix2pix']
ROOT_DIR = 'outputs/detection_agreement'
METRIC_TYPE = 'symmetric_nds'
#NVS_METRIC_TYPES = ["actor_psnr", "actor_ssim", "actor_lpips", "actor_coverage", "psnr", "ssim", "lpips", "scene_fid_all_cam", "scene_fid_all_cam_clip_vit_b_32", "scene_fid_per_cam", "scene_fid_per_cam_clip_vit_b_32"]
NVS_METRIC_TYPES = ['scene_fid_all_cam']

# If "agreement_vs_nvs_metric.csv" exists, load df from that file.
if os.path.exists('concat_agreement_vs_nvs_metric.csv'):
    df = pd.read_csv('concat_agreement_vs_nvs_metric.csv')

else:
    df_rows = []
    for model in MODELS:
        for aug in AUGS:
            agreement_results_path = f'{ROOT_DIR}/{model}/{AGGREGATION_METHOD}/{model}_detection_agreement_ft{aug}_thres05_{AGGREGATION_METHOD}_range_fraction_1.0_dist_ths_2.0.json'
            with open(agreement_results_path, 'r') as f:
                agreement_results = json.load(f)
            for sample_token, res in agreement_results.items():
                df_rows.append({
                    'model': model,
                    'aug': aug,
                    'metric_type': METRIC_TYPE,
                    'metric_value': res[METRIC_TYPE],
                    'sample_token': sample_token
                })

    df = pd.DataFrame(df_rows)
    for nvs_metric_type in NVS_METRIC_TYPES:
        df[nvs_metric_type] = np.nan
    df['scene'] = np.nan

    nvs_metric_pkl_path = 'clear_val_with_clipfid.pkl'
    with open(nvs_metric_pkl_path, 'rb') as f:
        nvs_metric = pickle.load(f)
    nvs_metric = pd.DataFrame.from_dict(nvs_metric)

    data_root_dir = '/home/s0001038/datasets/nuscenes'
    print('Loading NuScenes...')
    nusc = nuscenes.NuScenes(
        version='v1.0-trainval', dataroot=data_root_dir, verbose=False)
    print('Loaded NuScenes.')

    for i, row in df.iterrows():
        scene_name = nusc.get('scene', row['sample_token'])['name']
        curr_nvs = nvs_metric[nvs_metric['scene'] == scene_name]
        print('')
        for nvs_metric_type in NVS_METRIC_TYPES:
            row['scene'] = scene_name
            row[nvs_metric_type] = curr_nvs[nvs_metric_type].mean()
            df.iloc[i] = row

    df.to_csv('agreement_vs_nvs_metric.csv')

# Rename columns
new_nvs_metric_names = {
    'actor_psnr': 'Actor PSNR',
    'actor_ssim': 'Actor SSIM',
    'actor_lpips': 'Actor LPIPS',
    'psnr': 'PSNR',
    'ssim': 'SSIM',
    'lpips': 'LPIPS',
    'scene_fid_all_cam': 'FID',
    'scene_fid_all_cam_clip_vit_b_32': 'FID (CLIP)',
    'scene_fid_per_cam': 'FID per cam',
    'scene_fid_per_cam_clip_vit_b_32': 'FID (CLIP) per cam'
}
df.rename(columns=new_nvs_metric_names, inplace=True)
df.rename(columns={'aug': 'Augmentation'}, inplace=True)
new_aug_names = {
    '': 'Real only',
    np.nan: 'Real only',
    '_aug': 'Image augs',
    '_nerf': 'NeRF',
    '_pix2pix': 'Img2Img'
}
df['Augmentation'] = df['Augmentation'].replace(new_aug_names)

selected_metric_type = 'symmetric_nds'
fontsize = 18
sns.set_style('whitegrid')
palette = sns.color_palette('muted')[:len(AUGS)]
for model in MODELS:
    for image_metric in NVS_METRIC_TYPES:
        if image_metric == 'actor_coverage':
            continue
        plt.figure(figsize=(7, 7))
        for aug in AUGS:
            selected_data = df[(df['metric_type'] == selected_metric_type)
                               & (df['Augmentation'] == new_aug_names[aug])]
            correlation = selected_data[
                new_nvs_metric_names[image_metric]].corr(
                    selected_data['metric_value'])
            ax = sns.scatterplot(
                data=selected_data,
                x=new_nvs_metric_names[image_metric],
                y='metric_value',
                label=f'{new_aug_names[aug]}, corr.: {round(correlation, 2)}',
                s=75)
            sns.regplot(
                data=selected_data,
                x=new_nvs_metric_names[image_metric],
                y='metric_value',
                scatter=False,
                line_kws={'linewidth': 5})

        #ax.set_xlabel(new_nvs_metric_names[image_metric], fontsize=fontsize)
        ax.set_xlabel('')
        if image_metric == 'psnr' or image_metric == 'lpips':
            ax.set_ylabel('Detection agreement', fontsize=fontsize)
        else:
            ax.set_ylabel('')
        ax.set_ylim(0.4, 1)
        #plt.show()
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.savefig(
            f'outputs/detection_agreement/figs/{model}_detection_agreement_vs_{image_metric}_with_shifted_samples.pdf',
            bbox_inches='tight')
        print('Plotting done')
