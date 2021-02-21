import os
import torch
import json
import numpy as np
import sys
sys.path.append("/home/yichen/DenseNet")

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageOps, ImageFile

import cv2

from tqdm.autonotebook import tqdm

import datasets.vcoco.vsrl_utils as vu
from efficientdet.vcoco_dataset import *
from efficientdet.help_function import *


obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
           'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier','toothbrush']


class HICO_DET_Dataset(Dataset):
    def __init__(self, root_dir, set='train', transform=None, color_prob=0):
        # self.root_dir = root_dir
        self.data_dir = root_dir
        self.processed_dir = os.path.join(self.data_dir, "hico_processed")
        self.setname = set
        self.transform = transform
        self.color_prob = color_prob

        self.load_object_category()
        self.load_verb_category()
        self.load_hoi_category()
        self.load_ann_list()
        self.load_ann_by_image()

    def load_object_category(self):
        self.obj_to_id = {}
        self.id_to_obj = {}
        for id, obj in enumerate(obj_list):
            if obj != "":
                self.obj_to_id[obj] = id
                self.id_to_obj[id] = obj
        assert len(self.obj_to_id) == 80
        assert len(self.id_to_obj) == 80

    def load_verb_category(self):
        self.id_to_verb = {}
        self.verb_to_id = {}
        verb_list_path = os.path.join(self.processed_dir, "verb_list.json")
        with open(verb_list_path, "r") as file:
            verb_list = json.load(file)
        for item in verb_list:
            id = int(item["id"])
            name = item["name"]
            self.id_to_verb[id] = name
            self.verb_to_id[name] = id
        self.num_verbs = len(self.verb_to_id)

    def load_hoi_category(self):
        self.hoi_to_objid = {}
        self.hoi_to_verbid = {}
        hoi_list_path = os.path.join(self.processed_dir, "hoi_list.json")
        with open(hoi_list_path, "r") as file:
            hoi_list = json.load(file)
        for item in hoi_list:
            hoi_id = int(item["id"])
            object = item["object"]
            object = object.replace("_", " ")
            verb = item["verb"]
            self.hoi_to_objid[hoi_id] = self.obj_to_id[object]
            self.hoi_to_verbid[hoi_id] = self.verb_to_id[verb]
        self.num_hois = len(self.hoi_to_verbid)

    def load_ann_list(self):
        ann_list_path = os.path.join(self.processed_dir, "anno_list.json")
        with open(ann_list_path, "r") as file:
            ann_list = json.load(file)
        split_ann_list = []
        for item in ann_list:
            if self.setname in item["global_id"]:
                split_ann_list.append(item)
        self.split_ann_list = split_ann_list

    def load_ann_by_image(self):
        self.ann_by_image = []
        self.hoi_count = np.zeros(self.num_hois).tolist()
        self.verb_count = np.zeros(self.num_verbs).tolist()

        for image_id, image_item in enumerate(self.split_ann_list):
            img_anns = {}

            image_path_postfix = image_item["image_path_postfix"]
            img_path = os.path.join(self.data_dir, "images", image_path_postfix)
            img_anns["img_path"] = img_path

            hois = image_item["hois"]

            inters = []  # (human_bbox, object_bbox, object_category, [action_category])
            instances = []  # (instance_bbox, instance_category, [human_actions], [object_actions])

            for idx, hoi in enumerate(hois):
                id_to_inter = {}  # (human_id, object_id) : (human_bbox, object_bbox, object_category, [action_category])
                id_to_human = {}  # human_id: (instance_bbox, instance_category, [human_actions], [])
                id_to_object = {}  # object_id: (instance_bbox, instance_category, [object_actions])

                hoi_id = int(hoi["id"])
                if hoi["invis"]:
                    continue
                # print(len(hoi["connections"]), len(hoi["human_bboxes"]), len(hoi["object_bboxes"]))
                for i in range(len(hoi["connections"])):

                    connection = hoi["connections"][i]
                    human_bbox = hoi["human_bboxes"][connection[0]]
                    object_bbox = hoi["object_bboxes"][connection[1]]

                    inter_id = tuple([idx] + connection)
                    human_id = tuple([idx] + [connection[0]])
                    object_id = tuple([idx] + [connection[1]])

                    self.hoi_count[hoi_id - 1] += 1
                    self.verb_count[self.hoi_to_verbid[hoi_id]-1] += 1

                    if inter_id in id_to_inter:
                        # id_to_inter[inter_id][3].append(hoi_id)
                        id_to_inter[inter_id][3].append(self.hoi_to_verbid[hoi_id])

                    else:
                        item = []
                        item.append(human_bbox)
                        item.append(object_bbox)
                        item.append(self.hoi_to_objid[hoi_id])
                        item.append([self.hoi_to_verbid[hoi_id]])
                        # item.append([hoi_id])
                        id_to_inter[inter_id] = item

                    if human_id in id_to_human:
                        id_to_human[human_id][2].append(self.hoi_to_verbid[hoi_id])
                    else:
                        id_to_human[human_id] = [human_bbox, 0, [self.hoi_to_verbid[hoi_id]], []]

                    if object_id in id_to_object:
                        id_to_object[object_id][3].append(self.hoi_to_verbid[hoi_id])
                    else:
                        id_to_object[object_id] = [object_bbox, self.hoi_to_objid[hoi_id], [], [self.hoi_to_verbid[hoi_id]]]

                inters += list(id_to_inter.values())
                instances = instances + list(id_to_human.values()) + list(id_to_object.values())

            unique_instances = []
            for inst in instances:
                m = 0.7
                minst = None
                for uinst in unique_instances:
                    if inst[1] == uinst[1] and single_iou(inst[0], uinst[0]) > m:
                        minst = uinst
                        m = single_iou(inst[0], uinst[0])
                if minst is None:
                    unique_instances.append(inst)
                else:
                    minst[2] += inst[2]
                    minst[3] += inst[3]

            unique_inters = []
            for inter in inters:
                m = 0.7 ** 2
                minter = None
                for uinter in unique_inters:
                    hiou = single_iou(inter[0], uinter[0])
                    oiou = single_iou(inter[1], uinter[1])
                    if inter[2] == uinter[2] and hiou > 0.7 and oiou > 0.7 and hiou*oiou > m:
                        minter = uinter
                        m = hiou * oiou
                if minter is None:
                    unique_inters.append(inter)
                else:
                    minter[3] += inter[3]


            # human_instances = list(id_to_human.values())
            # obj_instances = []
            # for id, obj in id_to_object.items():
            #     if obj[1] == 0: # human, judge overlap with human instance
            #         flag = False
            #         for hinst in human_instances:
            #             if single_iou(hinst[0], obj[0]) > 0.75:
            #                 hinst[3].extend(obj[3])
            #                 flag = True
            #                 break
            #         if not flag:
            #             obj_instances.append(obj)
            # instances = human_instances + obj_instances

        #     if len(unique_instances) > 0:
        #         img_anns["interaction"] = unique_inters
        #         img_anns["instance"] = unique_instances
        #         self.ann_by_image.append(img_anns)
        #     else:
        #         no_inst += 1
        # print("%d images has no instances"%no_inst)
            img_anns["interaction"] = unique_inters
            img_anns["instance"] = unique_instances
            self.ann_by_image.append(img_anns)
        self.num_images = len(self.ann_by_image)
        # with open("hico-det_hoi_count.json", "w") as file:
        #     json.dump(self.hoi_count, file)
        # with open("hico-det_verb_count.json", "w") as file:
        #     json.dump(self.verb_count, file)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        img_item = self.ann_by_image[index]
        img = self.load_img(img_item["img_path"])

        annot_bbox = {"instance": [], "interaction": []}
        for i, ann in enumerate(img_item["instance"]):
            tmp = np.zeros(4 + 1 + self.num_verbs * 2)  # (bbox, obj_cat, human action, object action)
            tmp[0:4] = ann[0]  # bbox
            tmp[4] = ann[1]  # object category
            human_act = np.zeros(self.num_verbs)  # human action
            obj_act = np.zeros(self.num_verbs)   # object action

            h_acts = np.array(ann[2]) - 1
            o_acts = np.array(ann[3]) - 1

            if h_acts.shape[0] > 0:
                human_act[h_acts] = 1
            if o_acts.shape[0] > 0:
                obj_act[o_acts] = 1

            tmp[5:5+self.num_verbs] = human_act
            tmp[5+self.num_verbs:5+2*self.num_verbs] = obj_act
            annot_bbox["instance"].append(tmp)

        for i, ann in enumerate(img_item["interaction"]):
            # tmp = np.zeros(12 + 1 + self.num_hois)  # (human bbox, object bbox, union bbox, obj category, union action)
            tmp = np.zeros(12 + 1 + self.num_verbs)  # (human bbox, object bbox, union bbox, obj category, union action)
            tmp[0:4] = ann[0]
            tmp[4:8] = ann[1]
            tmp[8:12] = self.merge_bbox(ann[0], ann[1])
            tmp[12] = ann[2]

            union_acts = np.zeros(self.num_verbs)

            u_acts = np.array(ann[3]) - 1
            union_acts[u_acts] = 1
            tmp[13:] = union_acts
            annot_bbox["interaction"].append(tmp)

        for key in annot_bbox:
            annot_bbox[key] = np.array(annot_bbox[key])

        sample = {'img': img, 'annot': annot_bbox}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def merge_bbox(self, b1, b2):
        if b1[0] < 0:
            return b2
        if b2[0] < 0:
            return b1
        return [min(b1[0], b2[0]), min(b1[1], b2[1]),
                max(b1[2], b2[2]), max(b1[3], b2[3])]

    def load_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if np.random.uniform(0, 1) < self.color_prob:
            pil_img = Image.fromarray(img)
            img = np.array(randomColor(pil_img))
        return img.astype(np.float32) / 255.


if __name__=="__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms
    training_set = HICO_DET_Dataset(root_dir="/home/yichen/DenseNet/datasets", set="train",
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    training_params = {'batch_size': 4,
                       'shuffle': False,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': 0}
    training_generator = DataLoader(training_set, **training_params)


    # print("len:", len(training_generator))
    np.set_printoptions(precision=3, suppress=True, threshold=np.inf)

    for epoch in range(100):
        print("epoch:", epoch)
        progress_bar = tqdm(training_generator)
        for i, data in enumerate(training_generator):
            # if iter < step - last_epoch * num_iter_per_epoch:
            #     progress_bar.update()
            #     continue
            imgs = data['img']
            annot = data['annot']

            # for key in annot:
            #     print(key, annot[key].numpy())



