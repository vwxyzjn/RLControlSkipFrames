import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

plot_indices = [[0,0], [1,0],[0,1],[1,1]]
plot_idx = 0
all_experiment_data = []
fig, ax = plt.subplots(2, 2, figsize=(15,10))
for root, dirs, filenames in os.walk("outputs"):
    if len(dirs) == 0:
        experiment_name =  os.path.basename(root)
        experiment_data = {}
        for filename in filenames:
            seed, frame_skip = [int(item) for item in filename.split(".")[0:2]]
            file_path = os.path.join(root, filename)
            data = np.genfromtxt(file_path, delimiter=',')
            if frame_skip not in experiment_data:
                experiment_data[frame_skip] = [data]
            else:
                experiment_data[frame_skip] += [data]
        
        all_experiment_data += [experiment_data]
        for key in sorted(experiment_data.keys()):
            min_length = min([len(item) for item in experiment_data[key]])
            average_data = np.average(
                    np.array([item[:min_length] for item in experiment_data[key]]),
                    axis=0)
            smoothed_average_data = gaussian_filter1d(average_data, sigma=2)
            idx = plot_indices[plot_idx]
            ax[idx[0], idx[1]].plot(smoothed_average_data, label=f"frame_skip = {key}")
            ax[idx[0], idx[1]].set_title(experiment_name)
    
        plot_idx += 1
plt.legend()
plt.savefig("imgs/experiment_results.svg")
plt.show()