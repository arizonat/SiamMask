import cv2
import glob
import sys
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

folder1 = sys.argv[1]
folder2 = sys.argv[2]

ious1_file = join(folder1,"IOUs.npy")
ious2_file = join(folder2,"IOUs.npy")

iou1 = np.load(ious1_file)
iou2 = np.load(ious2_file)

num_objs = iou1.shape[0]
num_frames = iou1.shape[1]
frames = list(range(num_frames))

for obj_id in range(num_objs):
    plt.figure(obj_id)
    plt.plot(frames, iou1[obj_id].T,'r',label="ours")
    plt.plot(frames, iou2[obj_id].T,'b',label="original")
    plt.xlabel("frame")
    plt.ylabel("iou")
    plt.legend()
plt.show()
