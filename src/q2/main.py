import os
import cv2    
import numpy as np
from math import floor
from segment_graph import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def suppress_qt_warnings():
    os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"

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
    return ret

def comp2Mask(comp, djs, h, w):
    mask = np.zeros(shape=(h,w), dtype= np.uint8)
    hashed_pixs = djs.all_pix_in_comp(comp)
    for y in range(h):
        for x in range(w):
            if y * w + x in hashed_pixs:
                mask[y, x] = 255
    return mask

def calcFeatureMatrix(img, bin_num, djs):
    img_rgb_fvec = calcRgbHistFeature(img, bin_num)
    fmat = []
    for comp in djs.all_comp():
        mask = comp2Mask(comp, djs, img.shape[0], img.shape[1])
        comp_rgb_fvec = calcRgbHistFeature(img, bin_num, mask)
        fvec = np.concatenate((comp_rgb_fvec, img_rgb_fvec))
        fmat.append(fvec)
    fmat = np.array(fmat)
    return fmat

def calcLabelVec(ht, comps, gt_seg):
    y_train = []
    for comp in comps:
        y, x = ht.vertice2pix(comp)
        if gt_seg[y,x] == 255:
            y_train.append(1)
        else:
            y_train.append(0)
    return y_train

def make_blobs(im_path, gt_path, gt_seg_path):
    pic_list = os.listdir(im_path)
    k, sigma, min, bin_num, n_features = 80, 0.8, 20, 8, 50
    pca = PCA(n_components=n_features)
    data, label = [], []

    for (i,pic) in enumerate(pic_list):
        print(i,"/", len(pic_list))
        img, gt, gt_seg = cv2.imread(im_path+pic), cv2.imread(gt_path+pic), cv2.imread(gt_seg_path+pic)
        gt, gt_seg = cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY), cv2.cvtColor(gt_seg,cv2.COLOR_BGR2GRAY)

        ht = vp_hash_table(img.shape[0], img.shape[1])
        djs = segment(img, sigma, k, min)

        fmat = calcFeatureMatrix(img, bin_num, djs)
        for fvec in fmat:
            data.append(fvec)

        label = label + calcLabelVec(ht, djs.all_comp(), gt_seg)
    data = pca.fit_transform(data)
    data, label = np.array(data), np.array(label)
    return data, label

if __name__ == '__main__':  
    suppress_qt_warnings()
    im_path, test_im_path = "../../data/train/imgs/", "../../data/test/imgs/"
    gt_path, test_gt_path = "../../data/train/gt/", "../../data/test/gt/"
    gt_seg_path, test_gt_seg_path = "../../data/train/seg_gt/", "../../data/test/seg_gt/"
    
    print("data making")
    x_train, y_train = make_blobs(im_path, gt_path, gt_seg_path)
    print("train data done")
    x_test, y_test = make_blobs(test_im_path, test_gt_path, test_gt_seg_path)
    print("test data done")
    rfc = RandomForestClassifier()
    scores = cross_val_score(rfc, x_train, y_train)
    print('交叉验证准确率为:'+str(scores.mean()))

    rfc.fit(x_train, y_train)
    y_train_predict = rfc.predict(x_train)
    y_test_predict = rfc.predict(x_test)
    print("训练集上的准确率：", accuracy_score(y_train, y_train_predict))
    print("测试集上的准确率：", accuracy_score(y_test, y_test_predict))