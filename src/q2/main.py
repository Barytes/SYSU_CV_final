import os
import numpy as np 
from segment_graph import *
from data_generation import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def suppress_qt_warnings():
    os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"

def generate_data_file():
    im_path, test_im_path = "../../data/train/imgs/", "../../data/test/imgs/"
    gt_path, test_gt_path = "../../data/train/gt/", "../../data/test/gt/"
    gt_seg_path, test_gt_seg_path = "../../data/train/seg_gt/", "../../data/test/seg_gt/"
    train_file_path, test_file_path = "../../data/train/rgb_feature_data/", "../../data/test/rgb_feature_data/"
    print("training data making...")
    x_train, y_train = make_blobs(im_path, gt_seg_path)
    print("train data done")
    print("test data making...")
    x_test, y_test = make_blobs(test_im_path, test_gt_seg_path)
    print("test data done")
    np.save(train_file_path+"x_train.npy",x_train)
    np.save(train_file_path+"y_train.npy",y_train)
    np.save(test_file_path+"x_test.npy",x_test)
    np.save(test_file_path+"y_test.npy",y_test)

def classify():
    train_file_path, test_file_path = "../../data/train/rgb_feature_data/", "../../data/test/rgb_feature_data/"
    x_train, y_train = np.load(train_file_path+"x_train.npy"), np.load(train_file_path+"y_train.npy")
    x_test, y_test = np.load(test_file_path+"x_test.npy"), np.load(test_file_path+"y_test.npy")
    print(x_train[0:20,0])
    print(y_train[0:20])
    print("data loaded...\nstart classifying...")
    rfc = RandomForestClassifier()
    scores = cross_val_score(rfc, x_train, y_train)
    print('交叉验证准确率为:'+str(scores.mean()))

    rfc.fit(x_train, y_train)
    y_train_predict = rfc.predict(x_train)
    y_test_predict = rfc.predict(x_test)
    print("训练集上的准确率：", accuracy_score(y_train, y_train_predict))
    print("测试集上的准确率：", accuracy_score(y_test, y_test_predict))

    

if __name__ == '__main__':
    suppress_qt_warnings()
    classify()

    
