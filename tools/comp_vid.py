import cv2
import glob
import sys
from os.path import join

folder1 = sys.argv[1]
folder2 = sys.argv[2]

images1 = sorted(glob.glob(join(folder1,"[0-9]*.png")))
images2 = sorted(glob.glob(join(folder2,"[0-9]*.png")))

for image1_file, image2_file in zip(images1, images2):
    im1 = cv2.imread(image1_file)
    im2 = cv2.imread(image2_file)
    cv2.imshow("im1",im1)
    cv2.imshow("im2",im2)
    cv2.waitKey(0)
