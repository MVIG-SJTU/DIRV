from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import matplotlib.pyplot as plt

import pickle
import json
import numpy as np
import cv2
import os
import sys
import argparse

import matplotlib as mpl
mpl.use('Agg')


obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
           'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
           'toothbrush']

with open("/DATA1/Benchmark/hico_20160224_det/hico_processed/hoi_list.json", "r") as file:
    hois = json.load(file)
num_hois = len(hois)
union_action_list = {}
for i, item in enumerate(hois):
    union_action_list[i] = item["verb"] + "_" + item["object"]


def visual_hico(preds_inst, detection, image_id):
    output_dir = "vis/%d" % image_id
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dpi = 80

    im_file = "./datasets/hico_20160224_det/images/test2015/HICO_test2015_" + (str(image_id)).zfill(8) + ".jpg"

    im_data = plt.imread(im_file)
    height, width, nbands = im_data.shape
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, interpolation='nearest')

    for inst_id in range(len(preds_inst["rois"])):
        box = preds_inst["rois"][inst_id]
        ax.add_patch(
            plt.Rectangle((box[0], box[1]),
                          box[2] - box[0],
                          box[3] - box[1], fill=False,
                          edgecolor="orange", linewidth=3)
        )
        text = obj_list[preds_inst["obj_class_ids"][inst_id]] + " ," + "%.3f"%preds_inst["obj_scores"][inst_id]
        ax.text(box[0] + 10, box[1] + 25,
                text, fontsize=20, color='blue')
    fig.savefig(os.path.join(output_dir, "instances.jpg"))
    plt.close()

    for ele_id, ele in enumerate(detection):
        role_scores = ele[3]
        role_scores_idx_sort = np.argsort(role_scores)[::-1]

        if role_scores.max() < 1:
            continue

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(im_data, interpolation='nearest')

        H_box = ele[0]
        O_box = ele[1]

        ax.add_patch(
            plt.Rectangle((H_box[0], H_box[1]),
                          H_box[2] - H_box[0],
                          H_box[3] - H_box[1], fill=False,
                          edgecolor="red", linewidth=3)
        )

        ax.add_patch(
            plt.Rectangle((O_box[0], O_box[1]),
                          O_box[2] - O_box[0],
                          O_box[3] - O_box[1], fill=False,
                          edgecolor="blue", linewidth=3)
        )

        for action_count in range(5):
            text = union_action_list[role_scores_idx_sort[action_count]] + ", %.2f" % role_scores[role_scores_idx_sort[action_count]]
            ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 35,
                text, fontsize=16, color='green')

        fig.savefig(os.path.join(output_dir, "%d.jpg"%ele_id))

        plt.close()



if __name__=="__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--det_file', type=str, default=None)
    args = ap.parse_args()

    detection = pickle.load(open(args.det_file, "rb"))
    visualize(detection)
