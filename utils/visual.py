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


def visual(detection, image_id=None):
    if image_id is None:
        image_id = detection[0]["image_id"]

    cc = plt.get_cmap('hsv', lut=6)
    dpi = 80

    im_file = './datasets/vcoco/coco/images/val2014/COCO_val2014_' + (str(image_id)).zfill(12) + '.jpg'
    im_data = plt.imread(im_file)
    height, width, nbands = im_data.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, interpolation='nearest')

    HO_dic = {}
    HO_set = set()
    count = 0

    for ele in detection:
        if (ele['image_id'] == image_id):
            action_count = -1

            for action_key, action_value in ele.items():
                if (action_key.split('_')[-1] != 'agent') and action_key != 'image_id' and action_key != 'person_box':
                    if (not np.isnan(action_value[0])) and (action_value[4] > 0.01):
                        O_box = action_value[:4]
                        H_box = ele['person_box']

                        action_count += 1

                        if tuple(O_box) not in HO_set:
                            HO_dic[tuple(O_box)] = count
                            HO_set.add(tuple(O_box))
                            count += 1
                        if tuple(H_box) not in HO_set:
                            HO_dic[tuple(H_box)] = count
                            HO_set.add(tuple(H_box))
                            count += 1

                        ax.add_patch(
                            plt.Rectangle((H_box[0], H_box[1]),
                                          H_box[2] - H_box[0],
                                          H_box[3] - H_box[1], fill=False,
                                          edgecolor=cc(HO_dic[tuple(H_box)])[:3], linewidth=3)
                        )
                        text = action_key.split('_')[0] + ', ' + "%.2f" % action_value[4]

                        ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 35,
                                text,
                                bbox=dict(facecolor=cc(HO_dic[tuple(O_box)])[:3], alpha=0.5),
                                fontsize=16, color='white')

                        ax.add_patch(
                            plt.Rectangle((O_box[0], O_box[1]),
                                          O_box[2] - O_box[0],
                                          O_box[3] - O_box[1], fill=False,
                                          edgecolor=cc(HO_dic[tuple(O_box)])[:3], linewidth=3)
                        )
                        ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
    fig.savefig("vis/%d.jpg"%image_id)


def visual_demo(detection, im_path, save_path):
    cc = plt.get_cmap('hsv', lut=6)
    dpi = 80
    im_data = plt.imread(im_path)
    height, width, nbands = im_data.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, interpolation='nearest')

    HO_dic = {}
    HO_set = set()
    count = 0

    for ele in detection:
        action_count = -1
        for action_key, action_value in ele.items():
            if (action_key.split('_')[-1] != 'agent') and action_key != 'image_id' and action_key != 'person_box':
                if (not np.isnan(action_value[0])) and (action_value[4] > 0.01):
                    O_box = action_value[:4]
                    H_box = ele['person_box']
                    action_count += 1
                    if tuple(O_box) not in HO_set:
                        HO_dic[tuple(O_box)] = count
                        HO_set.add(tuple(O_box))
                        count += 1
                    if tuple(H_box) not in HO_set:
                        HO_dic[tuple(H_box)] = count
                        HO_set.add(tuple(H_box))
                        count += 1
                    ax.add_patch(
                        plt.Rectangle((H_box[0], H_box[1]),
                                      H_box[2] - H_box[0],
                                      H_box[3] - H_box[1], fill=False,
                                      edgecolor=cc(HO_dic[tuple(H_box)])[:3], linewidth=3)
                    )
                    text = action_key.split('_')[0] + ', ' + "%.2f" % action_value[4]
                    ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 35,
                            text,
                            bbox=dict(facecolor=cc(HO_dic[tuple(O_box)])[:3], alpha=0.5),
                            fontsize=16, color='white')
                    ax.add_patch(
                        plt.Rectangle((O_box[0], O_box[1]),
                                      O_box[2] - O_box[0],
                                      O_box[3] - O_box[1], fill=False,
                                      edgecolor=cc(HO_dic[tuple(O_box)])[:3], linewidth=3)
                    )
                    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
    fig.savefig(save_path)


if __name__=="__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--det_file', type=str, default=None)
    args = ap.parse_args()

    detection = pickle.load(open(args.det_file, "rb"))
    visualize(detection)
