import json

import matplotlib.pyplot as plt
import numpy as np

metrics_summary_path_real = 'metrics_summary_fcos_real.json'
metrics_summary_path_sim = 'metrics_summary_fcos_sim.json'

with open(metrics_summary_path_real,
          'rb') as f_real, open(metrics_summary_path_sim, 'rb') as f_sim:
    metrics_summary_real = json.load(f_real)
    metrics_summary_sim = json.load(f_sim)

classes = list(metrics_summary_real['mean_dist_aps'].keys())

fig, ax = plt.subplots()
bar_width = 0.35
opacity = 0.8
index = np.arange(len(classes))

real_values = [metrics_summary_real['mean_dist_aps'][cls] for cls in classes]
sim_values = [metrics_summary_sim['mean_dist_aps'][cls] for cls in classes]

bar1 = plt.bar(
    index - bar_width / 2,
    real_values,
    bar_width,
    alpha=opacity,
    color='b',
    label='real')
bar2 = plt.bar(
    index + bar_width / 2,
    sim_values,
    bar_width,
    alpha=opacity,
    color='g',
    label='sim')

plt.xlabel('class')
plt.ylabel('mAP')
plt.title('mAP for real and sim data')
plt.xticks(index, classes)
plt.legend()

plt.show()

# Calculate gap for each class and print the result as percentage
gap = [100 * (real - sim) / real for sim, real in zip(sim_values, real_values)]
for cls, g in zip(classes, gap):
    print(f'Gap for class {cls}: {g:.2f}%')
