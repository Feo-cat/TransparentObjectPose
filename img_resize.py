import cv2
import os

img_file = "/home/renchengwei/GDR-Net/test_repo/IMG_5044.JPG"
img = cv2.imread(img_file)
# anti-aliasing
img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
cv2.imwrite("/home/renchengwei/GDR-Net/test_repo/5044_640_480.png", img)