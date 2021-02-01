import os
import numpy as np 
import cv2
from sklearn.decomposition import PCA
from segment_graph import *
from data_generation import *
from sklearn.cluster import KMeans  
import matplotlib.pyplot as plt



def suppress_qt_warnings():
    os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"

def calcPatch(x, y, patch_size, img):
    patch = np.zeros(shape=(patch_size,patch_size,3), dtype = np.uint8)
    x0, y0 = x - 8, y - 7
    for i in range(16):
        for j in range(16):
                patch[i, j, :] = img[y0+i, x0+j, :]
    return patch

if __name__ == '__main__':  
    im_path, test_im_path = "../../data/train/imgs/", "../../data/test/imgs/"
    gt_path, test_gt_path = "../../data/train/gt/", "../../data/test/gt/"
    gt_seg_path, test_gt_seg_path = "../../data/train/seg_gt/", "../../data/test/seg_gt/"

    img = cv2.imread(test_im_path+"57.png") 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    patch_size, hist_bin_num = 16, 4
    rgbfmat = []

    sift=cv2.xfeatures2d.SIFT_create()
    kps, des = sift.detectAndCompute(gray, None) 
    kp_out_bound, patches = [], []
    for (i,kp) in enumerate(kps):
        y, x = round(kp.pt[0]), round(kp.pt[1]) #TODO:x?y?
        if y+8 >= img.shape[0] or x+7 >= img.shape[1]:
            kp_out_bound.append(i)
            continue
        if x-8 < 0 or y-7 < 0:
            kp_out_bound.append(i)
            continue
        patch = calcPatch(x, y, patch_size, img)
        patches.append(patch)
        rgbfvec = calcRgbHistFeature(patch, hist_bin_num)
        rgbfmat.append(rgbfvec)

    pca = PCA(n_components=10)
    reduced_des = pca.fit_transform(des)
    for j,k in enumerate(kp_out_bound):
        reduced_des = np.delete(reduced_des, k-j, 0)
    rgbfmat = np.array(rgbfmat)
    fmat = np.concatenate((reduced_des, rgbfmat), axis=1)
    
    kmeans=KMeans(n_clusters=3) 
    kmeans.fit(fmat)

    patches = np.array(patches)
    
    # for i,l in enumerate(kmeans.labels_):  
    #     plt.plot(x1[i],x2[i],color=colors[l],marker=markers[l],ls='None')  
    # plt.show() 