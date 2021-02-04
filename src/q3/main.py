import os
import numpy as np 
import cv2
from math import sqrt, ceil, floor
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

def paste_patch(figure, patch, patch_size, x0, y0):
    for i in range(patch_size):
        for j in range(patch_size):
            figure[y0+i, x0+j, :] = patch[i, j, :]

def empty_patch(patch_size):
    patch = np.zeros(shape=(patch_size, patch_size, 3))
    return patch

def print_patches(patches):
    for p in patches:
        cv2.imshow("", p)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def draw_sep_lines(fig, sorted_labels, n_clusters, n_w, patch_size):
    _,w,_ = fig.shape
    color, thickness = (0, 0, 255), 1
    n_kps = 0
    for c in range(n_clusters):
        n_kps += len([x for x in sorted_labels if x[1]==c])
        bottom_y,bottom_x = (floor(n_kps / n_w)+1)*patch_size, (n_kps % n_w)*patch_size
        if bottom_x == 0:
            bottom_x += n_clusters
        cv2.line(fig, (0,bottom_y), (bottom_x,bottom_y), color, thickness)
        cv2.line(fig, (bottom_x,bottom_y), (bottom_x,bottom_y-patch_size), color, thickness)
        cv2.line(fig, (bottom_x,bottom_y-patch_size), (w,bottom_y-patch_size), color, thickness)

if __name__ == '__main__':  
    im_path, test_im_path = "../../data/train/imgs/", "../../data/test/imgs/"
    gt_path, test_gt_path = "../../data/train/gt/", "../../data/test/gt/"
    gt_seg_path, test_gt_seg_path = "../../data/train/seg_gt/", "../../data/test/seg_gt/"
    output_path, test_output_path = "../../data/train/cluster_fig/", "../../data/test/cluster_fig/"

    pic_list = os.listdir(test_im_path)
    for p in pic_list:
        img = cv2.imread(test_im_path+p) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        patch_size, hist_bin_num, n_clusters = 16, 4, 3
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

        kmeans=KMeans(n_clusters=n_clusters) 
        kmeans.fit(fmat)

        sorted_labels = sorted(enumerate(kmeans.labels_), key=lambda x:x[1])

        n_patches = len(patches)
        cf_w = ceil(sqrt(float(n_patches)))
        cf_h = ceil(float(n_patches)/float(cf_w))
        n_empty_patches = cf_w*cf_h-n_patches

        for i in range(n_empty_patches):
            patches.append(empty_patch(patch_size))
            sorted_labels.append((n_patches,-1))
            n_patches += 1

        cluster_figure = np.zeros(shape=(cf_h*patch_size, cf_w*patch_size, 3))
        idx_patches = 0
        for i in range(cf_h):
            for j in range(cf_w):
                paste_patch(cluster_figure, patches[sorted_labels[idx_patches][0]], patch_size, j*patch_size, i*patch_size)
                idx_patches += 1

        draw_sep_lines(cluster_figure, sorted_labels, n_clusters, cf_w, patch_size)

        cv2.imwrite(test_output_path+p,cluster_figure)