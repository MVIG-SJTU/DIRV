import os
import torch
import numpy as np
import copy
import sys
sys.path.append("/home/yichen/DenseNet")

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageOps, ImageFile

from pycocotools.coco import COCO
import cv2
import random
from tqdm.autonotebook import tqdm

import datasets.vcoco.vsrl_utils as vu
from efficientdet.help_function import *


label_to_class = {0: ('hold', 'obj'), 1: ('sit', 'instr'), 2: ('ride', 'instr'), 3: ('look', 'obj'),
                  4: ('hit', 'instr'), 5: ('hit', 'obj'), 6: ('eat', 'obj'), 7: ('eat', 'instr'),
                  8: ('jump', 'instr'), 9: ('lay', 'instr'), 10: ('talk_on_phone', 'instr'),
                  11: ('carry', 'obj'), 12: ('throw', 'obj'), 13: ('catch', 'obj'), 14: ('cut', 'instr'),
                  15: ('cut', 'obj'), 16: ('work_on_computer', 'instr'), 17: ('ski', 'instr'),
                  18: ('surf', 'instr'), 19: ('skateboard', 'instr'), 20: ('drink', 'instr'),
                  21: ('kick', 'obj'), 22: ('point', 'instr'), 23: ('read', 'obj'), 24: ('snowboard', 'instr')}

sub_label_to_class = {0: 'hold', 1: 'stand', 2: 'sit', 3: 'ride', 4: 'walk', 5: 'look', 6: 'hit',
                       7: 'eat', 8: 'jump', 9: 'lay', 10: 'talk_on_phone', 11: 'carry', 12: 'throw',
                       13: 'catch', 14: 'cut', 15: 'run', 16: 'work_on_computer', 17: 'ski', 18: 'surf',
                       19: 'skateboard', 20: 'smile', 21: 'drink', 22: 'kick', 23: 'point', 24: 'read',
                       25: 'snowboard'}

obj_label_to_class = {26: ('hold', 'obj'), 27: ('sit', 'instr'), 28: ('ride', 'instr'), 29: ('look', 'obj'),
                      30: ('hit', 'instr'), 31: ('hit', 'obj'), 32: ('eat', 'obj'), 33: ('eat', 'instr'),
                      34: ('jump', 'instr'), 35: ('lay', 'instr'), 36: ('talk_on_phone', 'instr'),
                      37: ('carry', 'obj'), 38: ('throw', 'obj'), 39: ('catch', 'obj'), 40: ('cut', 'instr'),
                      41: ('cut', 'obj'), 42: ('work_on_computer', 'instr'), 43: ('ski', 'instr'),
                      44: ('surf', 'instr'), 45: ('skateboard', 'instr'), 46: ('drink', 'instr'),
                      47: ('kick', 'obj'), 48: ('point', 'instr'), 49: ('read', 'obj'), 50: ('snowboard', 'instr')}


class VCOCO_Dataset(Dataset):
    def __init__(self, root_dir, set='trainval', transform=None, color_prob=0):

        self.root_dir = root_dir
        self.setname = set
        self.transform = transform
        self.color_prob = color_prob

        self.coco = COCO(os.path.join(self.root_dir, "coco/annotations", "instances_trainval2014.json"))
        self.vcoco = vu.load_vcoco("vcoco_"+set, os.path.join(self.root_dir, "data"))

        self.image_ids = self.load_ids()
        self.load_classes()
        self.load_vcoco_classes()
        self.load_ann_by_image()

    def load_ids(self):
        ids_path = os.path.join(self.root_dir, "data/splits", "vcoco_%s.ids"%self.setname)
        with open(ids_path,"r") as f:
            ids = f.readlines()
        ids = [int(id) for id in ids]
        return ids


    def load_ann_by_image(self):
        self.ann_by_img = {}
        for i in range(len(self.vcoco)):
            for j in range(len(self.vcoco[i]["image_id"])):
                image_id = self.vcoco[i]["image_id"][j][0]
                if image_id not in self.ann_by_img:
                    self.ann_by_img[image_id] = {"subject":[], "object": [], "action": []} # subject/object/action ids
                if self.vcoco[i]["label"][j][0] > 0:
                    annot = self.ann_by_img[image_id]

                    for k, role in enumerate(self.vcoco[i]["role_name"][1:]):
                        annot["subject"].append(self.vcoco[i]["role_object_id"][j][0])
                        annot["object"].append(self.vcoco[i]["role_object_id"][j][k+1])
                        annot["action"].append(self.class_to_label[(self.vcoco[i]["action_name"], role)])
                    self.ann_by_img[image_id] = annot
        self.inter_ann_by_image = {}
        # instance: [ann_id, [sub_act_ids], [obj_act_ids]], interaction: [sub_ann_id, obj_ann_id, [act_ids]]
        for image_id in self.ann_by_img:
            instance_set = {}
            inter_set = {}
            annot = {"instance": [], "interaction": []}
            for i in range(len(self.ann_by_img[image_id]["subject"])):
                act_id = self.ann_by_img[image_id]["action"][i]
                sub_id = self.ann_by_img[image_id]["subject"][i]
                obj_id = self.ann_by_img[image_id]["object"][i]
                sub_act_id = self.actid_to_subactid(act_id)
                obj_act_id = self.actid_to_objactid(act_id)
                if sub_id > 0:
                    if sub_id not in instance_set:
                        instance_set[sub_id] = len(annot["instance"])
                        annot["instance"].append([sub_id, [sub_act_id], []])
                    else:
                        if act_id not in annot["instance"][instance_set[sub_id]][1]:
                            annot["instance"][instance_set[sub_id]][1].append(sub_act_id)
                if obj_id > 0:
                    if obj_id not in instance_set:
                        instance_set[obj_id] = len(annot["instance"])
                        annot["instance"].append([obj_id, [], [obj_act_id]])
                    else:
                        if act_id not in annot["instance"][instance_set[obj_id]][2]:
                            annot["instance"][instance_set[obj_id]][2].append(obj_act_id)

                if (sub_id, obj_id) not in inter_set:
                    inter_set[(sub_id, obj_id)] = len(annot["interaction"])
                    annot["interaction"].append([sub_id, obj_id, [act_id]])
                else:
                    annot["interaction"][inter_set[(sub_id, obj_id)]][2].append(act_id)

            # load other objects without interactions
            all_ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
            for ann_id in all_ann_ids:
                if ann_id not in instance_set:
                    instance_set[ann_id] = len(annot["instance"])
                    annot["instance"].append([ann_id, [], []])

            self.inter_ann_by_image[image_id] = annot

    def load_vcoco_classes(self):
        self.class_to_label = {}
        self.label_to_class = {}
        self.sub_class_to_label = {}
        self.sub_label_to_class = {}
        self.obj_class_to_label = {}
        self.obj_label_to_class = {}
        id = 0
        for i, item in enumerate(self.vcoco):
            self.sub_class_to_label[item["action_name"]] = i
            self.sub_label_to_class[i] = item["action_name"]
            for role in item["role_name"][1:]:
                self.class_to_label[(item["action_name"], role)] = id
                self.label_to_class[id] = (item["action_name"], role)
                self.obj_class_to_label[(item["action_name"], role)] = id + len(self.vcoco)
                self.obj_label_to_class[id + len(self.vcoco)] = (item["action_name"], role)
                id += 1
        # print(self.label_to_class)
        # print(self.sub_label_to_class)
        # print(self.obj_label_to_class)

    def actid_to_subactid(self, actid):
        act = self.label_to_class[actid]
        return self.sub_class_to_label[act[0]]

    def actid_to_objactid(self, actid):
        act = self.label_to_class[actid]
        return self.obj_class_to_label[act]

    def load_image(self, image_id):
        image_info = self.coco.loadImgs(image_id)[0]

        if self.setname == "test":
            coco_set = "val2014"
        else:
            coco_set = "train2014"

        path = os.path.join(self.root_dir, "coco/images/%s"%coco_set, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if np.random.uniform(0, 1) < self.color_prob:
            pil_img = Image.fromarray(img)
            img = np.array(randomColor(pil_img))

        return img.astype(np.float32) / 255.

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        image_id = self.image_ids[idx]

        img = self.load_image(image_id)
        annot = self.inter_ann_by_image[image_id]

        annot_bbox = {"instance": [], "interaction": []}
        for i, inst in enumerate(annot["instance"]):
            tmp = np.zeros(4+1+len(self.sub_label_to_class)+len(self.obj_class_to_label), dtype=np.float32)
            # (x1,y1,x2,y2, obj_class, sub_act_cls one hot, obj_act_cls one hot)
            tmp[:5] = self.id_to_bbox(inst[0])
            tmp[5:] = to_onehot(inst[1]+inst[2], len(self.sub_label_to_class)+len(self.obj_class_to_label))
            annot_bbox["instance"].append(tmp)

        for i, inter in enumerate(annot["interaction"]):
            tmp = np.zeros(12 + 1 + len(self.label_to_class), dtype=np.float32)  # (sub/obj/act bbox, obj_cls, act_cls one hot)
            bs = self.id_to_bbox(inter[0], need_category=True)
            bo = self.id_to_bbox(inter[1], need_category=True)
            bi = self.merge_bbox(bs[:4], bo[:4])
            tmp[:4] = bs[:4]
            tmp[4:8] = bo[:4]
            tmp[8:12] = bi
            tmp[12] = bo[4]
            tmp[13:] = to_onehot(inter[2], len(self.label_to_class))

            if bo[4] >= 0:
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


    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=image_index, iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_labels_inverse[a['category_id']]
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[c['id']-1] = c['id']
            self.coco_labels_inverse[c['id']] = c['id'] - 1
            self.classes[c['name']] = c['id'] - 1


        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def id_to_bbox(self, id, need_category=True):
        "From coco ann id to (x1,y1,x2,y2) bbox"
        if isinstance(id, int) or isinstance(id, np.int32) or isinstance(id, np.int64):
            if id > 0:
                bbox = self.coco.anns[id]["bbox"].copy()
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]
                bbox.append(self.coco_labels_inverse[self.coco.anns[id]["category_id"]])
            else:
                bbox = [-1.0, -1.0, -1.0, -1.0, -1.0]
            if not need_category:
                bbox = bbox[:4]
            return bbox
        elif isinstance(id, list) or isinstance(id, np.ndarray):
            bboxs = []
            for id_ in id:
                if id_ > 0:
                    bbox = self.coco.anns[id_]["bbox"].copy()
                    bbox[2] = bbox[0] + bbox[2]
                    bbox[3] = bbox[1] + bbox[3]
                    bbox.append(self.coco_labels_inverse[self.coco.anns[id_]["category_id"]])
                    bboxs.append(bbox)
                else:
                    bboxs.append([-1.0, -1.0, -1.0, -1.0, -1.0])
            bboxs = np.array(bboxs)
            if not need_category:
                bboxs = bboxs[:,:4]
            return bboxs
        else:
            print(type(id))
            raise(Exception, "id must be int or list")


def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    func = [color_enhance, brightness_enhance, contrast_enhance, sharpness_enchance]
    random.shuffle(func)
    for f in func:
        image = f(image)
    return image


def color_enhance(image):
    random_factor = np.random.uniform(0.8, 1.2)  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    return color_image


def brightness_enhance(image):
    random_factor = np.random.uniform(0.8, 1.2)  # 随机因子
    brightness_image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
    return brightness_image


def contrast_enhance(image):
    random_factor = np.random.uniform(0.8, 1.2)  # 随机因子
    contrast_image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
    return contrast_image


def sharpness_enchance(image):
    random_factor = np.random.uniform(0.8, 1.2)  # 随机因子
    sharp_image = ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
    return sharp_image


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5, crop_prob=1):
        image, annots = sample['img'], sample['annot']
        if np.random.rand() < flip_x:
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            for key in ["instance", "interaction"]:
                if len(annots[key]) == 0:
                    continue
                if key == "instance":
                    t = 1
                else:
                    t = 3
                for i in range(t):
                    w = annots[key][:, 4*i+2] - annots[key][:, 4*i+0]

                    annots[key][:, 4*i+2][annots[key][:, 4*i+2] > 0] = (cols - annots[key][:, 4*i+0])[annots[key][:, 4*i+2] > 0]
                    annots[key][:, 4*i+0][annots[key][:, 4*i+0] > 0] = (annots[key][:, 4*i+2] - w)[annots[key][:, 4*i+0] > 0]

        if np.random.rand() < crop_prob:
            raw_h = image.shape[0]
            raw_w = image.shape[1]

            if len(annots["interaction"]) > 0:
                xmin = np.min(annots["interaction"][:, 8])
                ymin = np.min(annots["interaction"][:, 9])
                xmax = np.max(annots["interaction"][:, 10])
                ymax = np.max(annots["interaction"][:, 11])
            else:
                xmin = raw_w
                ymin = raw_h
                xmax = 0
                ymax = 0

            if len(annots["instance"]) > 0:
                instance_area = (annots["instance"][:, 2] - annots["instance"][:, 0]) * (annots["instance"][:, 3] - annots["instance"][:, 1])

            xmin = min(xmin, raw_w - raw_w / 2)
            ymin = min(ymin, raw_h - raw_h / 2)
            xmax = max(xmax, raw_w / 2)
            ymax = max(ymax, raw_h / 2)

            x1 = int(np.random.uniform(0, xmin))
            y1 = int(np.random.uniform(0, ymin))

            x2 = int(np.random.uniform(max(xmax, x1+raw_w/2), raw_w))
            y2 = int(np.random.uniform(max(ymax, y1+raw_h/2), raw_h))
            # x2 = int(np.random.uniform(xmax, raw_w)) + 1
            # y2 = int(np.random.uniform(ymax, raw_h)) + 1

            if len(annots["interaction"]) > 0:
                annots["interaction"][:, [0, 2, 4, 6, 8, 10]] -= x1
                annots["interaction"][:, [1, 3, 5, 7, 9, 11]] -= y1
            if len(annots["instance"]) > 0:
                annots["instance"][:, [0, 2]] -= x1
                annots["instance"][:, [1, 3]] -= y1
                new_instance_area = (annots["instance"][:, 2] - annots["instance"][:, 0]) * (annots["instance"][:, 3] - annots["instance"][:, 1])
                remain_idx = (new_instance_area / instance_area) > 0.5
                annots["instance"] = annots["instance"][remain_idx]

            image = image[y1:y2, x1:x2, :]

        sample = {'img': image, 'annot': annots}

        return sample


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        for key in ["instance", "interaction"]:
            if len(annots[key]) == 0:
                continue
            if key == "instance":
                t = 4
            else:
                t = 12
            annots[key][:, :t] *= scale
            annots[key][:, :t][annots[key][:, :t] < 0] = -1

        for key in ["instance", "interaction"]:
            annots[key] = torch.from_numpy(np.array(annots[key]))

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': annots, 'scale': scale}


def collater(data):
    imgs = [s['img'] for s in data]
    annots = {}
    annots["instance"] = [s['annot']['instance'] for s in data]
    annots["interaction"] = [s['annot']['interaction'] for s in data]

    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = {
        "instance": max(instance.shape[0] for instance in annots["instance"]),
        "interaction": max(interaction.shape[0] for interaction in annots["interaction"])
    }

    annot_len = {
        "instance": 0,
        "interaction": 0
    }
    # annot_len = {
    #     "instance": 4 + 1 + len(sub_label_to_class) + len(obj_label_to_class),
    #     "interaction": 12 + 1 + len(label_to_class)
    # }
    for key in ["instance", "interaction"]:
        for item in annots[key]:
            if len(item.shape) > 1:
                annot_len[key] = item.shape[1]
                break

    annot_padded = {}

    for key in ["instance", "interaction"]:
        if max_num_annots[key] > 0:
            annot_padded[key] = torch.ones((len(annots[key]), max_num_annots[key], annot_len[key])) * -1
            for idx, annot in enumerate(annots[key]):
                if annot.shape[0] > 0:
                    annot_padded[key][idx, :annot.shape[0], :] = annot
        else:
            annot_padded[key] = torch.ones((len(annots[key]), 1, annot_len[key])) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}
    # annot: {instance: Tensor, interaction: Tensor}
    # instance: n*m*(obj bpx,obj_cls,sub_act_cls_one_hot,obj_act_cls_one_hot),
    # interaction: n*m*(sub/obj/act box, obj_cls, act_cls_one_hot)
    # n: batch size, m: max count in single image



def draw_bbox(imgs, annots):
    batch_size = len(imgs)
    for i in range(batch_size):
        img_np = imgs[i].permute(1, 2, 0).numpy() * 255
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_np = img_np.astype(np.uint8)

        img_inter = copy.deepcopy(img_np)
        for annot in annots["interaction"][i]:
            if annot[0] < 0:
                continue
            cv2.rectangle(img_inter, (annot[0], annot[1]), (annot[2], annot[3]), color=(255, 0, 0))
            cv2.rectangle(img_inter, (annot[4], annot[5]), (annot[6], annot[7]), color=(0, 255, 0))
            cv2.rectangle(img_inter, (annot[8], annot[9]), (annot[10], annot[11]), color=(0, 0, 255))
        cv2.imwrite("imgs/%d_inter.png"%i, img_inter)

        img_inst = copy.deepcopy(img_np)
        for annot in annots["instance"][i]:
            if annot[0] < 0:
                continue
            cv2.rectangle(img_inst, (annot[0], annot[1]), (annot[2], annot[3]), color=(255, 0, 0))

        cv2.imwrite("imgs/%d_inst.png" % i, img_inst)


if __name__=="__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms
    training_set = VCOCO_Dataset(root_dir="/home/yichen/DenseNet/datasets/vcoco", set="test",
                               transform=transforms.Compose([
                                    # Normalizer(),
                                    Augmenter(),
                                    Resizer()
                               ]))

    training_params = {'batch_size': 32,
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

            draw_bbox(imgs, annot)

            break
        break