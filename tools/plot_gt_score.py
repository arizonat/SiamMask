import numpy as np
import matplotlib.pyplot as plt
import glob
from os.path import join
import sys

results_dir = sys.argv[1]

results_files = glob.glob(join(results_dir,"*"))

fig, ax = plt.subplots()

for i, results_file in enumerate(results_files):
    print(results_file)
    gt_scores = np.load(join(results_file, "gt_scores.npy"))
    frames = list(range(gt_scores.shape[1]))
    ax.plot(frames, gt_scores.T)
    #ax.legend(results_file)
plt.show()
