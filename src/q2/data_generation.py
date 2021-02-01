import os
import cv2    
import numpy as np
from math import floor
from segment_graph import *
from disjoint_set import *
from sklearn.decomposition import PCA

def calcRgbHistFeature(image, bin_num, mask = None):
    if mask is None:
        height, width, _ = image.shape
        img = np.reshape(image, newshape = (height*width,3))
    else:
        img = reshapeMaskedImg(image, mask)
    length, channel = img.shape
    assert channel == 3                             
    interval = 256 / bin_num
    colorspace = np.zeros(shape = (bin_num, bin_num, bin_num), dtype = float)
    for p in range(length):
            pix_val = img[p,:]
            i, j, k = floor(pix_val[0]/interval), floor(pix_val[1]/interval), floor(pix_val[2]/interval)
            colorspace[i, j, k] += 1
    fvec = np.reshape(colorspace, newshape= bin_num ** 3)
    fvec = fvec / length
    return fvec

def reshapeMaskedImg(image, mask):
    assert image.shape[:2] == mask.shape
    front_size = len(mask[mask==255])
    ret = np.zeros(shape=(front_size, 3), dtype = np.uint8)
    h, w, _ = image.shape
    i = 0
    for r in range(h):
        for c in range(w):
            if mask[r,c] == 255:
                ret[i] = image[r,c]
                i += 1
    return ret

def comp2Mask(comp, djs, ht):
    mask = np.zeros(shape=(ht.h,ht.w), dtype= np.uint8)
    vertices = djs.all_vertices_in_comp(comp)
    for v in vertices:
        pix = ht.vertice2pix(v)
        mask[pix[0],pix[1]] = 255
    return mask

def calcFeatureMatrix(img, bin_num, djs, ht):
    img_rgb_fvec = calcRgbHistFeature(img, bin_num)
    fmat = []
    for comp in djs.all_comp():
        mask = comp2Mask(comp, djs, ht)
        comp_rgb_fvec = calcRgbHistFeature(img, bin_num, mask)
        fvec = np.concatenate((comp_rgb_fvec, img_rgb_fvec))
        fmat.append(fvec)
    fmat = np.array(fmat)
    return fmat

def calcLabelVec(ht, comps, gt_seg):
    y_train = []
    for comp in comps:
        (y, x) = ht.vertice2pix(comp)
        if gt_seg[y,x] == 255:
            y_train.append(1)
        else:
            y_train.append(0)
    return y_train

def make_blobs(im_path, gt_seg_path):
    pic_list = os.listdir(im_path)
    k, sigma, min, bin_num, n_features = 80, 0.8, 20, 8, 50
    pca = PCA(n_components=n_features)
    data, label = [], []

    for (i,pic) in enumerate(pic_list):
        print(i,"/", len(pic_list))
        img, gt_seg = cv2.imread(im_path+pic), cv2.imread(gt_seg_path+pic)
        gt_seg = cv2.cvtColor(gt_seg,cv2.COLOR_BGR2GRAY)

        djs = segment(img, sigma, k, min)
        ht = vp_hash_table(img.shape[0], img.shape[1])

        fmat = calcFeatureMatrix(img, bin_num, djs, ht)
        for fvec in fmat:
            data.append(fvec)

        label = label + calcLabelVec(ht, djs.all_comp(), gt_seg)
    data = pca.fit_transform(data)
    data, label = np.array(data), np.array(label)
    return data, label