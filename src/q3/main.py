import os
import numpy as np 
import cv2
from sklearn.decomposition import PCA
from segment_graph import *
from data_generation import *
from sklearn.cluster import KMeans  



def suppress_qt_warnings():
    os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"

def calcPatch(kp, patch_size, img):
    
    return None

if __name__ == '__main__':  
    im_path, test_im_path = "../../data/train/imgs/", "../../data/test/imgs/"
    gt_path, test_gt_path = "../../data/train/gt/", "../../data/test/gt/"
    gt_seg_path, test_gt_seg_path = "../../data/train/seg_gt/", "../../data/test/seg_gt/"

    img = cv2.imread(test_im_path+"57.png", cv2.IMREAD_GRAYSCALE) 
    patch_size, hist_bin_num = 16, 4
    rgbfmat = []

    sift=cv2.xfeatures2d.SIFT_create()
    kps, des = sift.detectAndCompute(img, None) 
    
    for kp in kps:
        x, y = round(kp.pt[0]), round(kp.pt[1]) #TODO:x?y?
        if kp.pt[0] > img.shape[0] or kp.pt[1] > img.shape[1]:
            print(kp.pt[0], kp.pt[1])
            continue
        patch = calcPatch(x, y, patch_size, img)
    #     rgbfvec = calcRgbHistFeature(patch, hist_bin_num)
    #     rgbfmat.append(rgbfvec)

    # pca = PCA(n_components=10)
    # reduced_des = pca.fit_transform(des)
    # rgbfmat = np.array(rgbfmat)
    # fmat = np.concatenate((reduced_des, rgbfmat), axis=1)

    # kmeans=KMeans(n_clusters=3)
    # kmeans.fit(fmat)
    # for i,l in enumerate(kmeans.labels_):  
    #     plt.plot(x1[i],x2[i],color=colors[l],marker=markers[l],ls='None')  
    # plt.show() 