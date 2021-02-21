# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/annotations/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os
import cv2
import time
import glob

import argparse
import torch
import yaml
import pickle
import numpy as np
# from pycocotools.cocoeval import COCOeval

# from utils.vsrl_eval import VCOCOeval
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from efficientdet.help_function import single_iou, single_ioa, single_inter, single_union
from utils.utils import preprocess, invert_affine, postprocess, postprocess_hoi, postprocess_dense_union, postprocess_hoi_flip, postprocess_dense_union_flip
# from utils.apply_prior import apply_prior
from utils.timer import Timer
from utils.visual_hico import visual_hico
from Generate_HICO_detection import Generate_HICO_detection


ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='hico-det', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=3, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.3, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=int, default=1)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=int, default=0)
ap.add_argument('--override', type=int, default=0, help='override previous bbox results file if exists')
ap.add_argument('--data_dir', type=str, default='./datasets', help='the root folder of dataset')
ap.add_argument('--need_visual', type=int, default=0, help='whether need to visualize the results')
ap.add_argument('--flip_test', type=int, default=1, help='whether apply flip augmentation when testing')


args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
need_visual = args.need_visual
weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights
data_dir = args.data_dir
project = args.project

params = yaml.safe_load(open(f'projects/{project}.yml'))
SET_NAME = params['val_set']
project_name = params["project_name"]


print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project}.yml'))
num_objects = 90
num_union_actions = 117
num_union_hois = 600
num_inst_actions = 234

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef]
output_dir = f"./logs/{project_name}/results"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if args.flip_test:
    detection_path = os.path.join(output_dir, f'{SET_NAME}_bbox_results_flip_final.pkl')
else:
    detection_path = os.path.join(output_dir, f'{SET_NAME}_bbox_results_final.pkl')

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

obj_dict = {}
cid = 0
for obj in obj_list:
    if obj != "":
        cid += 1
        obj_dict[obj] = cid

with open(args.data_dir + "/hico_20160224_det/hico_processed/verb_list.json", "r") as file:
    verbs_hico = json.load(file)
verbs_dict = {}
for id, item in enumerate(verbs_hico):
    verb_name = item["name"]
    verbs_dict[verb_name] = id

with open(args.data_dir + "/hico_20160224_det/hico_processed/hoi_list.json", "r") as file:
    hois_hico = json.load(file)
verb_to_hoi = {}
for hoi_id, item in enumerate(hois_hico):
    verb_id = verbs_dict[item["verb"]]
    if verb_id in verb_to_hoi:
        verb_to_hoi[verb_id].append(hoi_id)
    else:
        verb_to_hoi[verb_id] = [hoi_id]

n = 0
for verb_id in verb_to_hoi:
    n += len(verb_to_hoi[verb_id])
    verb_to_hoi[verb_id] = np.array(verb_to_hoi[verb_id])
assert n == num_union_hois


def calc_ioa(a, b):
    # a(anchor) [boxes, (x1, y1, x2, y2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    exp_x1 = np.expand_dims(a[:, 0], axis=1)
    exp_x2 = np.expand_dims(a[:, 2], axis=1)
    exp_y1 = np.expand_dims(a[:, 1], 1)
    exp_y2 = np.expand_dims(a[:, 3], 1)

    iw = np.where(exp_x2 < b[:, 2], exp_x2, b[:, 2]) - np.where(exp_x1 > b[:, 0], exp_x1, b[:, 0])
    ih = np.where(exp_y2 < b[:, 3], exp_y2, b[:, 3]) - np.where(exp_y1 > b[:, 1], exp_y1, b[:, 1])
    # iw = torch.clamp(iw, min=0)
    # ih = torch.clamp(ih, min=0)
    iw = np.where(iw > 0, iw, 0)
    ih = np.where(ih > 0, ih, 0)

    intersection = iw * ih
    area = np.where(area > 1e-6, area, 1e-6)
    IoA = intersection / area
    # IoA[torch.isnan(IoA)] = 1
    return IoA


def calc_iou(a, b):
    # a(anchor) [boxes, (x1, y1, x2, y2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    exp_x1 = np.expand_dims(a[:, 0], axis=1)
    exp_x2 = np.expand_dims(a[:, 2], axis=1)
    exp_y1 = np.expand_dims(a[:, 1], 1)
    exp_y2 = np.expand_dims(a[:, 3], 1)

    iw = np.where(exp_x2 < b[:, 2], exp_x2, b[:, 2]) - np.where(exp_x1 > b[:, 0], exp_x1, b[:, 0])
    ih = np.where(exp_y2 < b[:, 3], exp_y2, b[:, 3]) - np.where(exp_y1 > b[:, 1], exp_y1, b[:, 1])
    # iw = torch.clamp(iw, min=0)
    # ih = torch.clamp(ih, min=0)
    iw = np.where(iw > 0, iw, 0)
    ih = np.where(ih > 0, ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = np.where(ua > 0, ua, 1e-8)

    intersection = iw * ih
    IoU = intersection / ua
    return IoU


def transform_class_id(id):
    class_name = obj_list[id]
    hico_obj_id = obj_dict[class_name]
    return hico_obj_id


def transform_action_hico(act_scores, mode):
    union_scores = np.zeros(num_union_actions)
    for i in range(num_inst_actions//2):
        if mode == "subject":
            union_scores[verb_to_hoi[i]] = act_scores[i]
        else:
            union_scores[verb_to_hoi[i]] = act_scores[i + num_inst_actions//2]
    return union_scores


def xy_to_wh(bbox):
    ctr_x = (bbox[0] + bbox[2]) / 2
    ctr_y = (bbox[1] + bbox[3]) / 2
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return ctr_x, ctr_y, width, height


def fetch_location_score(anchor_bbox, obj_bbox, target_bbox, human_bbox, sigma):
    xo, yo, wo, ho = xy_to_wh(obj_bbox)
    xt, yt, wt, ht = xy_to_wh(target_bbox)
    # xh, yh, wh, hh = xy_to_wh(human_bbox)
    xa, ya, wa, ha = xy_to_wh(anchor_bbox)
    dist = np.zeros(4, dtype=np.float)
    dist[0] = (xo - xt) / wa
    dist[1] = (yo - yt) / ha
    # dist[0] = (xo - xt) / wh
    # dist[1] = (yo - yt) / hh
    # dist[2] = np.log(wo/wt)
    # dist[3] = np.log(ho/ht)

    return np.exp(-1*np.sum(dist**2)/(2*sigma**2))


def target_object_dist(target_objects_pos, objects_pos, anchors):
    width = anchors[:, 2] - anchors[:, 0]
    height = anchors[:, 3] - anchors[:, 1]
    anchors_size = np.stack([width, height], axis=1)
    anchors_size = np.expand_dims(anchors_size, axis=1)
    target_objects_pos = np.expand_dims(target_objects_pos, 1)
    diff = target_objects_pos - objects_pos
    diff = diff / anchors_size
    dist = np.sum(diff**2, axis=2)
    return dist


def hoi_match(image_id, preds_inst, preds_union, human_thre=0.3, anchor_thre=0.1, loc_thre=0.05):
    num_inst = len(preds_inst["rois"])
    humans = []
    objects = []
    human_bboxes = []
    human_inst_ids = []
    human_role_scores = []
    human_obj_scores = []

    while len(humans) == 0:
        if human_thre < 0.2:
            break
        for inst_id in range(num_inst):
            if preds_inst["obj_class_ids"][inst_id] != 0 or preds_inst["obj_scores"][inst_id] < human_thre:
                continue
            item = {}
            item["bbox"] = preds_inst["rois"][inst_id]
            item["role_scores"] = preds_inst["act_scores"][inst_id][:len(verb_to_hoi)]
            # item["role_scores"] = transform_action_hico(preds_inst["act_scores"][inst_id], "subject")
            item["obj_scores"] = preds_inst["obj_scores"][inst_id]
            item["inst_id"] = inst_id
            humans.append(item)
            human_bboxes.append(item["bbox"])
            human_inst_ids.append(item["inst_id"])
            human_role_scores.append(item["role_scores"])
            human_obj_scores.append(item["obj_scores"] )
        human_thre -= 0.1
    human_bboxes = np.array(human_bboxes)
    human_inst_ids = np.array(human_inst_ids)
    human_role_scores = np.array(human_role_scores)
    human_obj_scores = np.array(human_obj_scores)

    obj_role_scores = []
    obj_obj_scores = []
    for obj_id in range(len(preds_inst["rois"])):
        item = {}
        # obj_role_score = transform_action_hico(preds_inst["act_scores"][obj_id], "object")
        obj_role_score = preds_inst["act_scores"][obj_id][len(verb_to_hoi):]
        item["obj_role_scores"] = obj_role_score
        item["obj_scores"] = preds_inst["obj_scores"][obj_id]

        item["obj_class_id"] = preds_inst["obj_class_ids"][obj_id]

        obj_bbox = preds_inst["rois"][obj_id]
        item["bbox"] = obj_bbox
        objects.append(item)
        obj_role_scores.append(obj_role_score)
        obj_obj_scores.append(item["obj_scores"])
    object_bboxes = np.array(preds_inst["rois"])
    obj_role_scores = np.array(obj_role_scores)
    obj_obj_scores = np.array(obj_obj_scores)

    hoi_pair_score = np.zeros((len(humans), len(preds_inst["obj_class_ids"]), num_union_actions), dtype=np.float)

    if len(human_bboxes) > 0:
        IoA = calc_ioa(preds_union["rois"], human_bboxes)

        IoA_max = np.max(IoA, axis=1)
        human_foreground = IoA_max > 0.1  # 0.25
        human_IoA = IoA[human_foreground]
        for key in preds_union:
            preds_union[key] = preds_union[key][human_foreground]

        new_IoA = calc_ioa(preds_union["rois"], preds_inst["rois"])
        new_IoA_argmax = np.argmax(new_IoA, axis=1)
        new_IoA[np.arange(new_IoA.shape[0]), new_IoA_argmax] = 0
        new_IoA_sec_max = np.max(new_IoA, axis=1)
        obj_foreground = new_IoA_sec_max > 0.1  # 0.25
        for key in preds_union:
            preds_union[key] = preds_union[key][obj_foreground]

        human_IoU = calc_iou(preds_union["rois"], human_bboxes)
        human_IoA = human_IoA[obj_foreground]
        human_IoU_argmax = np.argmax(human_IoU * (human_IoA > 0.1), axis=1)  # 0.25
        obj_IoA = calc_ioa(preds_union["rois"], preds_inst["rois"])

        num_union = len(preds_union["rois"])
        num_human = len(human_bboxes)

        sp_vectors = preds_union["sp_vector"]
        inter_human_regions = human_bboxes[human_IoU_argmax]
        humans_pos_x = (inter_human_regions[:, 0] + inter_human_regions[:, 2]) / 2
        humans_pos_y = (inter_human_regions[:, 1] + inter_human_regions[:, 3]) / 2
        humans_pos = np.stack([humans_pos_x, humans_pos_y], axis=1)
        inter_objects_pos = humans_pos + sp_vectors

        objects_pos_x = (object_bboxes[:, 0] + object_bboxes[:, 2]) / 2
        objects_pos_y = (object_bboxes[:, 1] + object_bboxes[:, 3]) / 2
        objects_pos = np.stack([objects_pos_x, objects_pos_y], axis=1)

        obj_dists = target_object_dist(inter_objects_pos, objects_pos, preds_union["rois"])
        inter_human_instids = human_inst_ids[human_IoU_argmax]
        obj_dists[np.arange(num_union), inter_human_instids] = 100
        obj_dists[obj_IoA < 0.1] = 100  # 0.25
        inter_obj_ids = np.argmin(obj_dists, 1)
        inter_obj_dist = obj_dists[np.arange(num_union), inter_obj_ids]

        sigma = 0.6
        location_scores = np.exp(-1 * inter_obj_dist / (2 * sigma ** 2))
        location_scores = np.where(location_scores<loc_thre, 0, location_scores)
        anchor_scores = preds_union["act_scores"]
        anchor_scores = np.where(anchor_scores<anchor_thre, 0, anchor_scores)

        inter_human_ids = human_IoU_argmax
        inter_human_role_score = human_role_scores[inter_human_ids]
        inst_object_role_score = obj_role_scores[inter_obj_ids]

        # inter_human_obj_score = np.expand_dims(human_obj_scores[inter_human_ids],1)
        # inter_obj_obj_score = np.expand_dims(obj_obj_scores[inter_obj_ids],1)

        tau = 1.5
        # inter_scores = 0.5 * ((inter_human_role_score + inst_object_role_score) * anchor_scores).T * location_scores
        inter_scores = 0.5 * ((inter_human_role_score * inst_object_role_score) ** 0.5 * anchor_scores).T * location_scores ** tau

        inter_scores = inter_scores.T
        inter_scores[inst_object_role_score == 0] = 0

        for human_id in range(num_human):
            human_inter = inter_human_ids == human_id
            human_inter_obj_id = inter_obj_ids[human_inter]
            human_inter_score = inter_scores[human_inter]

            for obj_id in range(num_inst):
                hoi_pair_score[human_id, obj_id] = np.sum(human_inter_score[human_inter_obj_id==obj_id], axis=0)

    if args.flip_test:
        hoi_pair_score /= 2

    hoi_cat_pair_score = np.zeros((len(humans), len(preds_inst["obj_class_ids"]), num_union_hois), dtype=np.float)
    for verb in verb_to_hoi:
        hoi_cat_pair_score[:, :, verb_to_hoi[verb]] = hoi_pair_score[:, :, [verb]]

    dets = []
    for human_id, human in enumerate(humans):
        for obj_id, object in enumerate(objects):
            if human["inst_id"] == obj_id:
                continue

            tmp = []
            tmp.append(human["bbox"])  # human box
            tmp.append(object["bbox"])  # object box
            tmp.append(transform_class_id(object["obj_class_id"]))
            tmp.append(hoi_cat_pair_score[human_id, obj_id, :])
            tmp.append(human["obj_scores"])
            tmp.append(object["obj_scores"])
            dets.append(tmp)

    return dets


def img_detect(file, img_dir, model, input_size, regressBoxes, clipBoxes, threshold):
    fname, ext = os.path.splitext(file)
    image_id = int(fname.split("_")[-1])

    img_path = os.path.join(img_dir, file)

    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    if args.flip_test:
        ids = torch.arange(x.shape[-1]-1, -1, -1).long().cuda()
        x_flip = x[..., ids]
        x_cat = torch.cat([x, x_flip], 0)

    with torch.no_grad():
        if args.flip_test:
            features, union_act_cls, union_sub_reg, union_obj_reg, \
            inst_act_cls, inst_obj_cls, inst_bbox_reg, anchors = model(x_cat)

            anchors = torch.cat([anchors, anchors], 0)
            preds_union = postprocess_dense_union_flip(x_cat, anchors, union_act_cls, union_sub_reg, union_obj_reg,
                                                  regressBoxes, clipBoxes, 0.1, 1)
            preds_inst = postprocess_hoi_flip(x_cat, anchors, inst_bbox_reg, inst_obj_cls, inst_act_cls,
                                         regressBoxes, clipBoxes, threshold, nms_threshold,
                                         mode="object", classwise=True)
        else:
            features, union_act_cls, union_sub_reg, union_obj_reg, \
            inst_act_cls, inst_obj_cls, inst_bbox_reg, anchors = model(x)

            preds_union = postprocess_dense_union(x, anchors, union_act_cls, union_sub_reg, union_obj_reg,
                                            regressBoxes, clipBoxes, 0.1, 1)

            preds_inst = postprocess_hoi(x, anchors, inst_bbox_reg, inst_obj_cls, inst_act_cls,
                                     regressBoxes, clipBoxes, threshold, nms_threshold,
                                     mode="object", classwise=True)

        preds_inst = invert_affine(framed_metas, preds_inst)[0]
        preds_union = invert_affine(framed_metas, preds_union)[0]

        dets = hoi_match(image_id, preds_inst, preds_union)

    if need_visual:
        visual_hico(preds_inst, dets, image_id)
    return dets


def test(threshold=0.2):
    model = EfficientDetBackbone(num_classes=num_objects, num_union_classes=num_union_actions,
                                 num_inst_classes=num_inst_actions, compound_coef=args.compound_coef,
                                 ratios=eval(params["anchors_ratios"]), scales=eval(params["anchors_scales"]))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    if args.cuda:
        model = model.cuda()
    if args.float16:
        model = model.half()

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    img_dir = os.path.join(data_dir, "hico_20160224_det/images/%s" % "test2015")

    _t = {'im_detect': Timer(), 'misc': Timer()}
    detection = {}

    count = 0
    for line in glob.iglob(img_dir + '/' + '*.jpg'):
        count += 1

        _t['im_detect'].tic()
        image_id = int(line[-9:-4])

        file = "HICO_test2015_" + (str(image_id)).zfill(8) + ".jpg"

        # if file != "COCO_val2014_000000001987.jpg":
        #     continue

        dets = img_detect(file, img_dir, model, input_size, regressBoxes, clipBoxes, threshold=threshold)

        detection[image_id] = dets
        # detection.extend(img_detection)
        _t['im_detect'].toc()

        print('im_detect: {:d}/{:d}, average time: {:.3f}s'.format(count, 9658, _t['im_detect'].average_time))

    with open(detection_path, "wb") as file:
        pickle.dump(detection, file)


if __name__ == '__main__':
    if override_prev_results or not os.path.exists(detection_path):
        test()
    if args.flip_test:
        hico_dir = os.path.join(output_dir, f"{project_name}_flip_final/")
    else:
        hico_dir = os.path.join(output_dir, f"{project_name}_final/")
    if not os.path.exists(hico_dir):
        os.mkdir(hico_dir)
    Generate_HICO_detection(detection_path, hico_dir)




