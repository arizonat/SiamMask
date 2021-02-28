import sys
import glob
from os.path import isdir, join
from os import makedirs
import time
import cv2
import numpy as np

videofile = sys.argv[1]
dir1 = sys.argv[2]
dir2 = sys.argv[3]
outdir = sys.argv[4]

cap = cv2.VideoCapture(videofile)

mask_files1 = sorted(glob.glob(join(dir1,"*.png")))
mask_files2 = sorted(glob.glob(join(dir2,"*.png")))

res, img_raw = cap.read()
f = 0

if not isdir(outdir): makedirs(outdir)

while True:
    res, img_raw = cap.read()

    if not res:
        break
    
    mask1 = (cv2.imread(mask_files1[f])[:,:,0]>0).astype(np.uint8)
    mask2 = (cv2.imread(mask_files2[f])[:,:,0]>0).astype(np.uint8)

    img = img_raw.copy()

    _, cnt1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    pgon1 = cnt1[0]#.reshape(-1,2)
    box1 = np.int0(cv2.boxPoints(cv2.minAreaRect(pgon1)))

    _, cnt2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pgon2 = cnt2[0]#.reshape(-1,2)
    box2 = np.int0(cv2.boxPoints(cv2.minAreaRect(pgon2)))

    #img = cv2.drawContours(img, [box2], 0, (0,255,0), 1)
    #img = cv2.drawContours(img, [box1], 0, (255,0,0), 2)
    
    cv2.imwrite(join(outdir,"%05d.png"%(f)), img)
    
    cv2.imshow("img",img)
    cv2.waitKey(1)
    time.sleep(1/60)

    f = f + 1
