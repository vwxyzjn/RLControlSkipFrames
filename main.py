import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

all_experiment_data = []
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
            plt.plot(smoothed_average_data, label=f"frame_skip = {key}")
        plt.legend(bbox_to_anchor=(0, 1),
           bbox_transform=plt.gcf().transFigure)
        plt.suptitle(experiment_name)
        plt.savefig(f"imgs/{experiment_name}.svg")
        plt.show()
            
                