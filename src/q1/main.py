import imageio
import matplotlib.pyplot as plt
from filter import *
from segment_graph import *
import time
import os


# --------------------------------------------------------------------------------
# Segment an image:
# Returns a color image representing the segmentation.
#
# Inputs:
#           in_image: image to segment.
#           sigma: to smooth the image.
#           k: constant for threshold function.
#           min_size: minimum component size (enforced by post-processing stage).
#
# Returns:
#           num_ccs: number of connected components in the segmentation.
# --------------------------------------------------------------------------------
def segment(in_image, sigma, k, min_size):
    start_time = time.time()
    height, width, band = in_image.shape
    # print("Height:  " + str(height))
    # print("Width:   " + str(width))
    smooth_red_band = smooth(in_image[:, :, 0], sigma)
    smooth_green_band = smooth(in_image[:, :, 1], sigma)
    smooth_blue_band = smooth(in_image[:, :, 2], sigma)

    # build graph
    num, edges = build_graph(width, height, smooth_red_band, smooth_green_band, smooth_blue_band)
    
    # Segment
    u = segment_graph(width * height, num, edges, k)

    # post process small components
    for i in range(num):
        a = u.find(edges[i, 0])
        b = u.find(edges[i, 1])
        if (a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)):
            u.join(a, b)

    num_cc = u.num_sets()
    # output = random_coloring(u)

    # elapsed_time = time.time() - start_time
    # print(
    #     "Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
    #         int(elapsed_time % 60)) + " seconds")

    # displaying the result
    # fig = plt.figure()
    # a = fig.add_subplot(1, 2, 1)
    # plt.imshow(in_image)
    # a.set_title('Original Image')
    # a = fig.add_subplot(1, 2, 2)
    # plt.imshow(output)
    # a.set_title('Segmented Image')
    # plt.show()
    return num_cc, u

def build_graph(width, height, red, green, blue):
    # build graph
    edges_size = width * height * 4
    edges = np.zeros(shape=(edges_size, 3), dtype=object)
    num = 0
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int(y * width + (x + 1))
                edges[num, 2] = diff(red, green, blue, x, y, x + 1, y)
                num += 1
            if y < height - 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y + 1) * width + x)
                edges[num, 2] = diff(red, green, blue, x, y, x, y + 1)
                num += 1

            if (x < width - 1) and (y < height - 2):
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y + 1) * width + (x + 1))
                edges[num, 2] = diff(red, green, blue, x, y, x + 1, y + 1)
                num += 1

            if (x < width - 1) and (y > 0):
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y - 1) * width + (x + 1))
                edges[num, 2] = diff(red, green, blue, x, y, x + 1, y - 1)
                num += 1
    return num, edges

def random_coloring(u):
    output = np.zeros(shape=(height, width, 3), dtype = int)
    # pick random colors for each component
    colors = np.zeros(shape=(height * width, 3))
    for i in range(height * width):
        colors[i, :] = random_rgb()

    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            output[y, x, :] = colors[comp, :]
    return output

def suppress_qt_warnings():
    os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"

def select_k(sigma, min, input_path):
    # k = 80
    suppress_qt_warnings()
    pic_list = os.listdir(input_path)
    
    for k in range(80,84,1):
        cnt = 0
        for i in pic_list[0:100]:
            # Loading the image
            input_image = imageio.imread(input_path+i)
            # print("Loading is done.")
            # print("processing...")
            n,_,_ = segment(input_image, sigma, k, min)
            if n < 50 or n > 100:
                cnt += 1
        print(k, "\t", cnt)

def seg_all(k, sigma, min, input_path, output_path):
    pic_list = os.listdir(input_path)
    for i in pic_list:
        print("Processing ", i)
        input_image = imageio.imread(input_path+i)
        _,_,seg_image = segment(input_image, sigma, k, min)
        imageio.imwrite(output_path+i, seg_image)
        print(i, " done")

def mark_comp(k, sigma, min, input_path, gt_input_path):
    input_image = imageio.imread(input_path)
    gt_image = imageio.imread(gt_input_path)

    nc,u = segment(input_image, sigma, k, min)

    height, width, _ = input_image.shape
    comps = u.all_comp()
    comp_cnt = [0 for i in range(nc)]
    gt_comp = [0 for i in range(nc)]

    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            if (gt_image[y,x,:]==[255,255,255]).all():
                comp_cnt[comps.index(comp)] += 1

    for i in range(nc):
        if 2 * comp_cnt[i] >= u.size(comps[i]):
            gt_comp[i] = 1
    
    gt_seged = np.zeros(shape=(height , width, 3))
    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            if gt_comp[comps.index(comp)] == 1:
                gt_seged[y, x, :] = [255,255,255]
    return gt_seged

def mark_comp_all(k, sigma, min, input_path, gt_input_path, output_path):
    pic_list = os.listdir(input_path)
    for i in pic_list:
        print("Processing ", i)
        seg_image = mark_comp(k, sigma, min, input_path+i, gt_input_path+i)
        imageio.imwrite(output_path+i, seg_image)
        print(i, " done")

def calc_iou(gt_path, gt_seg_path, output_path):
    pic_list = os.listdir(gt_path)
    of = open(output_path+"iou.txt", 'w')
    of.write("pic\tr1&r2\tr1|r2\tiou\n")
    for i in pic_list:
        print("Processing ", i)
        gt,gt_seg = imageio.imread(gt_path+i),imageio.imread(gt_seg_path+i)
        height, width, _ = gt.shape
        r1nr2, r1or2 = 0.0, 0.0
        for y in range(height):
            for x in range(width):
                if (gt[y,x,:]==[255,255,255]).all() and (gt_seg[y,x,:]==[255,255,255]).all():
                    r1nr2 += 1
                if (gt[y,x,:]==[255,255,255]).all() or (gt_seg[y,x,:]==[255,255,255]).all():
                    r1or2 += 1
        iou = r1nr2/r1or2
        of.write(i+'\t'+str(r1nr2)+'\t'+str(r1or2)+'\t'+str(iou)+'\n')
        print(i, "done")
    of.close()

if __name__ == "__main__":
    suppress_qt_warnings()
    sigma = 0.8
    k = 80
    min = 20
    gt_path = "../../data/train/gt/"
    gt_seg_path = "../../data/train/seg_gt/"
    output_path = "../../data/train/"
    
    calc_iou(gt_path, gt_seg_path, output_path)
    

