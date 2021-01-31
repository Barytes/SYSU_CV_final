import os
import numpy as np 
import cv2
from segment_graph import *
from data_generation import *


def suppress_qt_warnings():
    os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"

if __name__ == '__main__':  
    im_path, test_im_path = "../../data/train/imgs/", "../../data/test/imgs/"
    gt_path, test_gt_path = "../../data/train/gt/", "../../data/test/gt/"
    gt_seg_path, test_gt_seg_path = "../../data/train/seg_gt/", "../../data/test/seg_gt/"
    img = cv2.imread(test_im_path+"57.png") 

    rows,cols = img.shape[:2]
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift=cv2.xfeatures2d.SIFT_create()
 
    kp, des = sift.detectAndCompute(gray, None) 
    print(kp.shape)
    print(des.shape)