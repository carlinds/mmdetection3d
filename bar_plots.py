import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#Data                  & Finetuning  & Augmentation  mAP   & NDS

data = [
    #['Real', 'No'   , 'FCOS3D'       , 'None'         , 32.2 , 39.7],
    ['Real', 'Yes', 'FCOS3D', 'None', 32.2, 39.8],
    ['Real', 'Yes', 'FCOS3D', 'Algorithmic', 32.5, 40.0],
    ['Real', 'Yes', 'FCOS3D', 'NeRF', 31.2, 38.6],
    ['Real', 'Yes', 'FCOS3D', 'Pix2Pix', 32.5, 39.8],
    #['Sim' , 'No'   , 'FCOS3D'       , 'None'         , 13.6 , 28.8],
    ['Sim', 'Yes', 'FCOS3D', 'None', 13.5, 28.8],
    ['Sim', 'Yes', 'FCOS3D', 'Algorithmic', 13.5, 28.9],
    ['Sim', 'Yes', 'FCOS3D', 'NeRF', 23.5, 33.6],
    ['Sim', 'Yes', 'FCOS3D', 'Pix2Pix', 24.5, 34.3],
    #['Real', 'No'   , 'PETR'         , 'None'         , 38.3 , 43.2],
    ['Real', 'Yes', 'PETR', 'None', 38.6, 43.1],
    ['Real', 'Yes', 'PETR', 'Algorithmic', 34.0, 38.9],
    ['Real', 'Yes', 'PETR', 'NeRF', 35.1, 40.0],
    ['Real', 'Yes', 'PETR', 'Pix2Pix', 31.4, 37.2],
    #['Sim',  'No'   , 'PETR'         , 'None'         , 18.8 , 30.5],
    ['Sim', 'Yes', 'PETR', 'None', 20.2, 31.6],
    ['Sim', 'Yes', 'PETR', 'Algorithmic', 20.4, 30.0],
    ['Sim', 'Yes', 'PETR', 'NeRF', 29.3, 37.3],
    ['Sim', 'Yes', 'PETR', 'Pix2Pix', 26.1, 35.1],
    #['Real', 'No'   , 'BEVFormer'    , 'None'         , 36.8 , 47.2],
    ['Real', 'Yes', 'BEVFormer', 'None', 38.4, 48.5],
    ['Real', 'Yes', 'BEVFormer', 'Algorithmic', 38.9, 48.6],
    ['Real', 'Yes', 'BEVFormer', 'NeRF', 38.5, 48.3],
    #['Real', 'Yes'  , 'BEVFormer'    , 'Pix2Pix'      , 36.9 , 47.2],
    ['Real', 'Yes', 'BEVFormer', 'Pix2Pix', 37.5, 48.1],
    #['Sim',  'No'   , 'BEVFormer'    , 'None'         , 28.0 , 41.8],
    ['Sim', 'Yes', 'BEVFormer', 'None', 29.1, 42.7],
    ['Sim', 'Yes', 'BEVFormer', 'Algorithmic', 31.0, 44.0],
    ['Sim', 'Yes', 'BEVFormer', 'NeRF', 31.7, 44.5],
    #['Sim',  'Yes'  , 'BEVFormer'    , 'Pix2Pix'      , 32.8 , 44.8],
    ['Sim', 'Yes', 'BEVFormer', 'Pix2Pix', 33.0, 44.9],
]
'Real only',
'Real only',
'Real only',
'Image augs',
'Image augs',
'Image augs',
'NeRF',
'NeRF',
'NeRF',
'Img2Img',
'Img2Img',
'Img2Img',

latex_table_data = [
    ['Real', 'FCOS3D', 'Real only', 32.2, 39.8, 100.0],
    ['Sim', 'FCOS3D', 'Real only', 13.5, 28.8, 46.3],
    #["Gap",  "FCOS3D", "Real only"  ,   58.1,   27.6,   53.7 ],
    ['Real', 'FCOS3D', 'Image augs', 32.5, 40.0, 100.0],
    ['Sim', 'FCOS3D', 'Image augs', 13.5, 28.9, 46.5],
    #["Gap",  "FCOS3D", "Image augs" ,   58.1,   27.4,   53.5 ],
    ['Real', 'FCOS3D', 'NeRF', 31.2, 38.6, 100.0],
    ['Sim', 'FCOS3D', 'NeRF', 23.5, 33.6, 58.7],
    #["Gap",  "FCOS3D", "NeRF"       ,   27.0,   15.6,   41.3 ],
    ['Real', 'FCOS3D', 'Img2Img', 32.5, 39.8, 100.0],
    ['Sim', 'FCOS3D', 'Img2Img', 24.5, 34.3, 57.3],
    #["Gap",  "FCOS3D", "Img2Img"    ,    23.9,   13.8,   42.7 ],
    ['Real', 'PETR', 'Real only', 38.6, 43.1, 100.0],
    ['Sim', 'PETR', 'Real only', 20.2, 31.6, 55.4],
    #["Gap",  "PETR",   "Real only"  ,   47.7,   26.7,   44.6 ],
    ['Real', 'PETR', 'Image augs', 34.0, 38.9, 100.0],
    ['Sim', 'PETR', 'Image augs', 20.4, 30.0, 57.6],
    #["Gap",  "PETR",   "Image augs" ,   47.2,   30.4,   42.4 ],
    ['Real', 'PETR', 'NeRF', 35.1, 40.0, 100.0],
    ['Sim', 'PETR', 'NeRF', 29.3, 37.3, 70.7],
    #["Gap",  "PETR",   "NeRF"       ,   24.1,   13.5,   29.3 ],
    ['Real', 'PETR', 'Img2Img', 31.4, 37.2, 100.0],
    ['Sim', 'PETR', 'Img2Img', 26.1, 35.1, 67.9],
    #["Gap",  "PETR",   "Img2Img"    ,    32.4,   18.6,   32.1 ],
    ['Real', 'BEVFormer', 'Real only', 38.4, 48.5, 100.0],
    ['Sim', 'BEVFormer', 'Real only', 29.1, 42.7, 76.6],
    #["Gap",  "BEVFormer", "Real only"  , 24.2,   12.0,   23.4 ],
    ['Real', 'BEVFormer', 'Image augs', 38.9, 48.6, 100.0],
    ['Sim', 'BEVFormer', 'Image augs', 31.0, 44.0, 77.6],
    #["Gap",  "BEVFormer", "Image augs" , 19.3,   9.3,    22.4 ],
    ['Real', 'BEVFormer', 'NeRF', 38.5, 48.3, 100.0],
    ['Sim', 'BEVFormer', 'NeRF', 31.7, 44.5, 78.9],
    #["Gap",  "BEVFormer", "NeRF"       , 17.4,   8.2,    21.1 ],
    ['Real', 'BEVFormer', 'Img2Img', 37.5, 48.1, 100.0],
    ['Sim', 'BEVFormer', 'Img2Img', 33.0, 44.9, 80.7],
    #["Gap",  "BEVFormer", "Img2Img"    ,  14.1,   7.4,    19.3 ],
]

df = pd.DataFrame(
    latex_table_data,
    columns=[
        'Evaluation data', 'Model', 'Fine-tuning method', 'mAP', 'NDS',
        'Detection Agreement'
    ])

fig, axes = plt.subplots(3, 3, figsize=(13, 13))
fontsize = 12
sns.set_style('whitegrid')
#palette = sns.color_palette("muted")
for j, metric_type in enumerate(['mAP', 'NDS', 'Detection Agreement']):
    # Create subplots

    # Create bar plots
    for i, model in enumerate(df['Model'].unique()):
        ax = sns.barplot(
            ax=axes[j, i],
            data=df[(df['Model'] == model)],
            x='Fine-tuning method',
            y=metric_type,
            hue='Evaluation data',
            ci=None)

        if j == 0:
            ax.set_title(model, fontdict={'fontsize': fontsize})

        for p in ax.patches:
            ax.annotate(
                format(p.get_height(), '.1f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=10,
                color='black')

        ax.set_xlabel('')
        if i == 0:
            ax.set_ylabel(metric_type, fontdict={'fontsize': fontsize})
            y_lim = df[(df['Model'] == model)][metric_type].max() + 15
        else:
            ax.tick_params(left=False)
            ax.set_yticks([])
            ax.set_ylabel('')

        ax.set_ylim(0, y_lim)
        if i == 0 and j == 0:  #i == len(df['model'].unique()) - 1:
            ax.legend(title='', loc='upper left', labels=['Real', 'Sim'])
        else:
            ax.legend_.remove()

        # Hide borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

plt.subplots_adjust(wspace=10, hspace=100)
plt.tight_layout()
plt.savefig('outputs/detection_agreement/figs/bar_plots.pdf')

print('hello')
# # Add legend for colors
# colors_legend = sns.color_palette("Set1", 4)
# colors_labels = ['Color 1', 'Color 2', 'Color 3', 'Color 4']
# color_handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors_legend]
# plt.legend(color_handles, colors_labels, loc='upper left', title='Color')

# # Add patterns
# patterns = ['/', '\\', '|', '-']
# for i, bar in enumerate(ax.patches):
#     hatching = patterns[i % len(patterns)]
#     bar.set_hatch(hatching)

# # Add legend for patterns
# patterns_legend = ['Pattern 1', 'Pattern 2', 'Pattern 3', 'Pattern 4']
# pattern_handles = [plt.Rectangle((0,0),1,1, hatch=patterns[i]) for i in range(len(patterns))]
# plt.legend(pattern_handles, patterns_legend, loc='upper right', title='Pattern')

# # Add labels
# ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '', 'A', 'B', 'C', 'D', 'E', 'F'])
# ax.set_xticks([1.5, 5.5, 9.5, 13.5, 17.5, 21.5])
# ax.set_xlabel('Groups')
# ax.set_ylabel('Values')
# ax.set_title('Bar Plot with Multiple Groups')

# plt.show()
