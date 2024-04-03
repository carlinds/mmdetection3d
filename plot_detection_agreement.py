import json
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODELS = ['fcos3d', 'petr', 'bevformer']
AUGS = ['', '_aug', '_nerf', '_pix2pix']
METRIC_TYPES = ['symmetric_nds']
AGGREGATION_METHODS = ['all']  #, "per_scene", "per_sample"]
RANGE_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for aggregation_method in AGGREGATION_METHODS:
    print(aggregation_method)
    #plot_results = {}
    df_rows = []
    for model in MODELS:
        #plot_results[model] = {}
        for aug in AUGS:
            #plot_results[model][aug] = {}
            #plot_results[model][aug]["range_fraction"] = []
            #for metric_type in METRIC_TYPES:
            #plot_results[model][aug][metric_type] = []

            for range_fraction in RANGE_FRACTIONS:
                if aggregation_method == 'per_sample':
                    result_path = f'outputs/detection_agreement/{model}/{aggregation_method}/{model}_detection_agreement_ft{aug}_thres05_range_fraction_{range_fraction}_dist_ths_2.0.json'
                else:
                    result_path = f'outputs/detection_agreement/{model}/{aggregation_method}/{model}_detection_agreement_ft{aug}_thres05_{aggregation_method}_range_fraction_{range_fraction}_dist_ths_2.0.json'

                with open(result_path, 'r') as f:
                    result = json.load(f)

                #plot_results[model][aug]["range_fraction"].append(range_fraction)
                for metric_type in METRIC_TYPES:
                    if aggregation_method == 'all':
                        result_value = result[metric_type]
                    else:
                        result_value = 0
                        for k, v in result.items():
                            if not math.isnan(v[metric_type]):
                                result_value += v[metric_type]
                        result_value /= len(result)

                    #plot_results[model][aug][metric_type].append(result_value)
                    if aug == '':
                        aug_str = 'Real only'
                    elif aug == '_aug':
                        aug_str = 'Image augs'
                    elif aug == '_nerf':
                        aug_str = 'NeRF'
                    elif aug == '_pix2pix':
                        aug_str = 'Img2Img'

                    if model == 'fcos3d':
                        model_str = 'FCOS3D'
                    elif model == 'petr':
                        model_str = 'PETR'
                    elif model == 'bevformer':
                        model_str = 'BEVFormer'

                    df_rows.append({
                        'Model': model_str,
                        'Augmentation': aug_str,
                        'range_fraction': range_fraction,
                        'metric_type': metric_type,
                        'metric_value': result_value
                    })
    df = pd.DataFrame(df_rows)

    for model in MODELS:
        if model == 'fcos3d':
            model_str = 'FCOS3D'
        elif model == 'petr':
            model_str = 'PETR'
        elif model == 'bevformer':
            model_str = 'BEVFormer'

        plt.figure(figsize=(6, 6))
        fontsize = 16
        sns.set_style('whitegrid')
        palette = sns.color_palette('muted')[:len(AUGS)]
        selected_metric_type = 'symmetric_nds'
        selected_data = df[(df['metric_type'] == selected_metric_type)
                           & (df['Model'] == model_str)]
        #ax = sns.lineplot(selected_data, x="range_fraction", y="metric_value", hue="Augmentation", style="Model", palette=palette, lw=3.5, markers=True, markersize=10)
        ax = sns.lineplot(
            selected_data,
            x='range_fraction',
            y='metric_value',
            hue='Augmentation',
            style='Augmentation',
            palette=palette,
            lw=3.5)
        #ax.set_title("Detection agreement vs. range fraction")
        ax.set_xlabel('Range fraction', fontsize=fontsize)
        ax.set_ylabel('Detection agreement', fontsize=fontsize)
        ax.set_ylim(0.4, 1)
        ax.set_xlim(0.1, 1)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.legend(fontsize=fontsize)
        #plt.show()
        plt.savefig(
            f'outputs/detection_agreement/figs/{model}_{aggregation_method}_detection_agreement_vs_range_fraction.pdf'
        )
        print('Plotting done')
