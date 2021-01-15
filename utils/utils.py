# Author: Zylo117

import os

import cv2
import numpy as np
import torch
from glob import glob
from torch import nn
from torchvision.ops import nms
from torchvision.ops.boxes import batched_nms
from typing import Union
import uuid

from utils.sync_batchnorm import SynchronizedBatchNorm2d

from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
import math


# obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#             'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#             'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
#             'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#             'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
#             'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
#             'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
#             'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#             'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#             'toothbrush']


def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,


def preprocess(*image_path, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def preprocess_video(*frame_from_video, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = frame_from_video
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh.sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            scores_, classes_ = classification_per[:, anchors_nms_idx].max(dim=0)
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out


def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])


def replace_w_sync_bn(m):
    for var_name in dir(m):
        target_attr = getattr(m, var_name)
        if type(target_attr) == torch.nn.BatchNorm2d:
            num_features = target_attr.num_features
            eps = target_attr.eps
            momentum = target_attr.momentum
            affine = target_attr.affine

            # get parameters
            running_mean = target_attr.running_mean
            running_var = target_attr.running_var
            if affine:
                weight = target_attr.weight
                bias = target_attr.bias

            setattr(m, var_name,
                    SynchronizedBatchNorm2d(num_features, eps, momentum, affine))

            target_attr = getattr(m, var_name)
            # set parameters
            target_attr.running_mean = running_mean
            target_attr.running_var = running_var
            if affine:
                target_attr.weight = weight
                target_attr.bias = bias

    for var_name, children in m.named_children():
        replace_w_sync_bn(children)


class CustomDataParallel(nn.DataParallel):
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, num_gpus):
        super().__init__(module)
        self.num_gpus = num_gpus

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus

        return [[inputs[i][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True)
                 for i in range(len(inputs))]
                for device_idx in range(len(devices))], \
               [kwargs] * len(devices)


def get_last_weights(weights_path):
    weights_path = glob(weights_path + f'/*.pth')
    weights_path = sorted(weights_path,
                          key=lambda x: int(x.rsplit('_')[-1].rsplit('.')[0]),
                          reverse=True)[0]
    print(f'using weights {weights_path}')
    return weights_path


def init_weights(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            if "conv_list" or "header" in name:
                variance_scaling_(module.weight.data)
            else:
                nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                if "classifier.header" in name:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    torch.nn.init.constant_(module.bias, bias_value)
                else:
                    module.bias.data.zero_()


def variance_scaling_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)


def postprocess_hoi(x, anchors, regression, obj_cls, act_cls, regressBoxes, clipBoxes, threshold, iou_threshold, mode="action", classwise=True):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    if mode == "action":
        main_cls = act_cls   # (bn, num_anchor, num_cls)
        other_cls = obj_cls  # (bn, num_anchor, num_cls)
    else:
        main_cls = obj_cls
        other_cls = act_cls
    scores = torch.max(main_cls, dim=2, keepdim=True)[0]  # (bn, num_anchor, 1)
    scores_over_thresh = (scores > threshold)[:, :, 0]  # (bn, num_anchor)
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh.sum() == 0:
            out.append({
                'rois': np.array(()),
                # 'act_class_ids': np.array(()),
                'act_scores': np.array(()),
                'obj_class_ids': np.array(()),
                'obj_scores': np.array(())
            })
            continue

        main_cls_per = main_cls[i, scores_over_thresh[i, :], ...].permute(1, 0)  # (num_cls, num_bbox)
        other_cls_per = other_cls[i, scores_over_thresh[i, :], ...].permute(1, 0)  # (num_cls, num_bbox)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...] # (num_bbox, 4)
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        if classwise:
            scores_, classes_ = main_cls_per.max(dim=0)
            anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)
        else:
            anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            main_scores_ = main_cls_per[:, anchors_nms_idx]  # (num_cls, num_nms_bbox)
            # main_scores_, main_classes_ = main_cls_per[:, anchors_nms_idx]  # (num_cls, num_nms_bbox)
            other_scores_ = other_cls_per[:, anchors_nms_idx]  # (num_cls, num_nms_bbox)
            # other_scores_, other_classes_ = other_cls_per[:, anchors_nms_idx]  # (num_cls, num_nms_bbox)
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]  # (num_nms_bbox, 4)

            if mode == "action":
                # act_classes_ = main_classes_.permute(1, 0)  # (num_nms_bbox, num_cls)
                act_scores_ = main_scores_.permute(1, 0)  # (num_nms_bbox, num_cls)
                # obj_classes_ = other_classes_.max(dim=0)  # (num_nms_bbox)
                obj_scores_, obj_classes_ = other_scores_.max(dim=0)  # (num_nms_bbox)
                # obj_classes_ = other_classes_.permute(1, 0)  # (num_nms_bbox, num_cls)
                # obj_scores_ = other_scores_.permute(1, 0)  # (num_nms_bbox, num_cls)
            else:
                # act_classes_ = other_classes_.permute(1, 0)
                act_scores_ = other_scores_.permute(1, 0)
                # obj_classes_ = main_classes_.max(dim=0)

                # arg_sort = torch.argsort(-1*main_scores_, 0)
                # for i in range(arg_sort.shape[1]):
                #     print("object category")
                #     for j in range(5):
                #         print(obj_list[arg_sort[j, i]], main_scores_[arg_sort[j,i], i])

                obj_scores_, obj_classes_ = main_scores_.max(dim=0)
                # obj_classes_ = main_classes_.permute(1, 0)
                # obj_scores_ = main_scores_.permute(1, 0)

            out.append({
                'rois': boxes_.cpu().numpy(),
                # 'act_class_ids': act_classes_.cpu().numpy(),
                'act_scores': act_scores_.cpu().numpy(),
                'obj_class_ids': obj_classes_.cpu().numpy(),
                'obj_scores': obj_scores_.cpu().numpy()
            })
        else:
            out.append({
                'rois': np.array(()),
                # 'act_class_ids': np.array(()),
                'act_scores': np.array(()),
                'obj_class_ids': np.array(()),
                'obj_scores': np.array(())
            })

    return out


def postprocess_dense(x, anchors, classification, clipBoxes, threshold, iou_threshold):
    transformed_anchors = torch.zeros_like(anchors).cuda()
    transformed_anchors[:, :, 0] = anchors[:, :, 1]
    transformed_anchors[:, :, 1] = anchors[:, :, 0]
    transformed_anchors[:, :, 2] = anchors[:, :, 3]
    transformed_anchors[:, :, 3] = anchors[:, :, 2]

    transformed_anchors = clipBoxes(transformed_anchors, x)

    main_cls = classification   # (bn, num_anchor, num_cls)

    scores = torch.max(main_cls, dim=2, keepdim=True)[0]  # (bn, num_anchor, 1)
    scores_over_thresh = (scores > threshold)[:, :, 0]  # (bn, num_anchor)
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh.sum() == 0:
            out.append({
                'rois': np.array(()),
                'act_class_ids': np.array(()),
                'act_scores': np.array(()),
            })

        main_cls_per = main_cls[i, scores_over_thresh[i, :], ...].permute(1, 0)  # (num_cls, num_bbox)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...] # (num_bbox, 4)
        scores_per = scores[i, scores_over_thresh[i, :], ...]

        if main_cls_per.shape[1] > 0:
            main_scores_ = main_cls_per[:, :]  # (num_cls, num_nms_bbox)
            boxes_ = transformed_anchors_per[:, :]  # (num_nms_bbox, 4)

            act_scores_ = main_scores_.permute(1, 0)  # (num_nms_bbox, num_cls)
            act_classes_ = main_scores_.max(dim=0)[1]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'act_class_ids': act_classes_.cpu().numpy(),
                'act_scores': act_scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'act_class_ids': np.array(()),
                'act_scores': np.array(()),
            })

    return out


def postprocess_dense_union(x, anchors, classification, sub_regression, obj_regression, regressBoxes, clipBoxes, threshold, iou_threshold=1, classwise=False):
    transformed_anchors = torch.zeros_like(anchors).cuda()
    transformed_anchors[:, :, 0] = anchors[:, :, 1]
    transformed_anchors[:, :, 1] = anchors[:, :, 0]
    transformed_anchors[:, :, 2] = anchors[:, :, 3]
    transformed_anchors[:, :, 3] = anchors[:, :, 2]

    transformed_anchors = clipBoxes(transformed_anchors, x)

    transformed_anchors_sub = regressBoxes(anchors, sub_regression)
    transformed_anchors_sub = clipBoxes(transformed_anchors_sub, x)

    transformed_anchors_obj = regressBoxes(anchors, obj_regression)
    transformed_anchors_obj = clipBoxes(transformed_anchors_obj, x)

    main_cls = classification   # (bn, num_anchor, num_cls)
    # other_cls = obj_classication

    scores = torch.max(main_cls, dim=2, keepdim=True)[0]  # (bn, num_anchor, 1)
    scores_over_thresh = (scores > threshold)[:, :, 0]  # (bn, num_anchor)
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh.sum() == 0:
            out.append({
                'rois': np.array(()),
                'rois_sub': np.array(()),
                'rois_obj': np.array(()),
                'sp_vector': np.array(()),
                'act_class_ids': np.array(()),
                'act_scores': np.array(()),
                # 'obj_scores': np.array(()),
            })
            continue

        main_cls_per = main_cls[i, scores_over_thresh[i, :], ...].permute(1, 0)  # (num_cls, num_bbox)
        # other_cls_per = other_cls[i, scores_over_thresh[i, :], ...].permute(1, 0)  # (num_cls, num_bbox)

        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...] # (num_bbox, 4)
        transformed_anchors_sub_per = transformed_anchors_sub[i, scores_over_thresh[i, :], ...] # (num_bbox, 4)
        transformed_anchors_obj_per = transformed_anchors_obj[i, scores_over_thresh[i, :], ...] # (num_bbox, 4)

        scores_per = scores[i, scores_over_thresh[i, :], ...]

        if iou_threshold < 1:
            if classwise:
                scores_, classes_ = main_cls_per.max(dim=0)
                anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)
            else:
                anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)
        else:
            anchors_nms_idx = np.arange(main_cls_per.shape[1])

        if anchors_nms_idx.shape[0] > 0:
            main_scores_ = main_cls_per[:, anchors_nms_idx]  # (num_cls, num_nms_bbox)
            # other_scores_ = other_cls_per[:, :]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]  # (num_nms_bbox, 4)
            boxes_sub_ = transformed_anchors_sub_per[anchors_nms_idx, :]  # (num_nms_bbox, 4)
            boxes_obj_ = transformed_anchors_obj_per[anchors_nms_idx, :]  # (num_nms_bbox, 4)
            sp_vector_x = (boxes_obj_[:, 0] + boxes_obj_[:, 2]) / 2 - (boxes_sub_[:, 0] + boxes_sub_[:, 2]) / 2
            sp_vector_y = (boxes_obj_[:, 1] + boxes_obj_[:, 3]) / 2 - (boxes_sub_[:, 1] + boxes_sub_[:, 3]) / 2

            sp_vector_x = sp_vector_x.reshape(-1, 1)
            sp_vector_y = sp_vector_y.reshape(-1, 1)

            sp_vector = torch.cat([sp_vector_x, sp_vector_y], 1)

            act_scores_ = main_scores_.permute(1, 0)  # (num_nms_bbox, num_cls)
            act_classes_ = main_scores_.max(dim=0)[1]
            # obj_scores_ = other_scores_.permute(1, 0)  #

            out.append({
                'rois': boxes_.cpu().numpy(),
                'rois_sub': boxes_sub_.cpu().numpy(),
                'rois_obj': boxes_obj_.cpu().numpy(),
                'sp_vector': sp_vector.cpu().numpy(),
                'act_class_ids': act_classes_.cpu().numpy(),
                'act_scores': act_scores_.cpu().numpy(),
                # 'obj_scores': obj_scores_.cpu().numpy()
            })
        else:
            out.append({
                'rois': np.array(()),
                'rois_sub': np.array(()),
                'rois_obj': np.array(()),
                'sp_vector': np.array(()),
                'act_class_ids': np.array(()),
                'act_scores': np.array(()),
                # 'obj_scores': np.array(())
            })

    return out


def postprocess_hoi_flip(x, anchors, regression, obj_cls, act_cls, regressBoxes, clipBoxes, threshold, iou_threshold, mode="action", classwise=True):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)

    if mode == "action":
        main_cls = act_cls   # (bn, num_anchor, num_cls)
        other_cls = obj_cls  # (bn, num_anchor, num_cls)
    else:
        main_cls = obj_cls
        other_cls = act_cls
    scores = torch.max(main_cls, dim=2, keepdim=True)[0]  # (bn, num_anchor, 1)
    scores_over_thresh = (scores > threshold)[:, :, 0]  # (bn, num_anchor)
    out = []
    n = x.shape[0] // 2
    for i in range(n):
        if scores_over_thresh.sum() == 0:
            out.append({
                'rois': np.array(()),
                # 'act_class_ids': np.array(()),
                'act_scores': np.array(()),
                'obj_class_ids': np.array(()),
                'obj_scores': np.array(())
            })
            continue

        main_cls_per = torch.cat([main_cls[i, scores_over_thresh[i, :], ...].permute(1, 0),
                                   main_cls[i+n, scores_over_thresh[i+n, :], ...].permute(1, 0)], 1)  # (num_cls, num_bbox)
        other_cls_per = torch.cat([other_cls[i, scores_over_thresh[i, :], ...].permute(1, 0),
                                    other_cls[i+n, scores_over_thresh[i+n, :], ...].permute(1, 0)], 1)  # (num_cls, num_bbox)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...] # (num_bbox, 4)
        transformed_anchors_per_flip = transformed_anchors[i+n, scores_over_thresh[i+n, :], ...].clone()

        cols = x.shape[3]
        w = transformed_anchors_per_flip[:, 2] - transformed_anchors_per_flip[:, 0]
        transformed_anchors_per_flip[:, 2] = cols - transformed_anchors_per_flip[:, 0]
        transformed_anchors_per_flip[:, 0] = transformed_anchors_per_flip[:, 2] - w

        transformed_anchors_per = torch.cat([transformed_anchors_per, transformed_anchors_per_flip], 0)

        scores_per = torch.cat([scores[i, scores_over_thresh[i, :], ...], scores[i+n, scores_over_thresh[i+n, :], ...]], 0)

        if classwise:
            scores_, classes_ = main_cls_per.max(dim=0)
            anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)
        else:
            anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            main_scores_ = main_cls_per[:, anchors_nms_idx]  # (num_cls, num_nms_bbox)
            # main_scores_, main_classes_ = main_cls_per[:, anchors_nms_idx]  # (num_cls, num_nms_bbox)
            other_scores_ = other_cls_per[:, anchors_nms_idx]  # (num_cls, num_nms_bbox)
            # other_scores_, other_classes_ = other_cls_per[:, anchors_nms_idx]  # (num_cls, num_nms_bbox)
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]  # (num_nms_bbox, 4)

            if mode == "action":
                # act_classes_ = main_classes_.permute(1, 0)  # (num_nms_bbox, num_cls)
                act_scores_ = main_scores_.permute(1, 0)  # (num_nms_bbox, num_cls)
                # obj_classes_ = other_classes_.max(dim=0)  # (num_nms_bbox)
                obj_scores_, obj_classes_ = other_scores_.max(dim=0)  # (num_nms_bbox)
                # obj_classes_ = other_classes_.permute(1, 0)  # (num_nms_bbox, num_cls)
                # obj_scores_ = other_scores_.permute(1, 0)  # (num_nms_bbox, num_cls)
            else:
                # act_classes_ = other_classes_.permute(1, 0)
                act_scores_ = other_scores_.permute(1, 0)
                # obj_classes_ = main_classes_.max(dim=0)
                obj_scores_, obj_classes_ = main_scores_.max(dim=0)
                # obj_classes_ = main_classes_.permute(1, 0)
                # obj_scores_ = main_scores_.permute(1, 0)

            out.append({
                'rois': boxes_.cpu().numpy(),
                # 'act_class_ids': act_classes_.cpu().numpy(),
                'act_scores': act_scores_.cpu().numpy(),
                'obj_class_ids': obj_classes_.cpu().numpy(),
                'obj_scores': obj_scores_.cpu().numpy()
            })
        else:
            out.append({
                'rois': np.array(()),
                # 'act_class_ids': np.array(()),
                'act_scores': np.array(()),
                'obj_class_ids': np.array(()),
                'obj_scores': np.array(())
            })

    return out


def postprocess_dense_union_flip(x, anchors, classification, sub_regression, obj_regression, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = torch.zeros_like(anchors).cuda()
    transformed_anchors[:, :, 0] = anchors[:, :, 1]
    transformed_anchors[:, :, 1] = anchors[:, :, 0]
    transformed_anchors[:, :, 2] = anchors[:, :, 3]
    transformed_anchors[:, :, 3] = anchors[:, :, 2]

    transformed_anchors = clipBoxes(transformed_anchors, x)

    transformed_anchors_sub = regressBoxes(anchors, sub_regression)
    transformed_anchors_sub = clipBoxes(transformed_anchors_sub, x)

    transformed_anchors_obj = regressBoxes(anchors, obj_regression)
    transformed_anchors_obj = clipBoxes(transformed_anchors_obj, x)

    main_cls = classification   # (bn, num_anchor, num_cls)

    scores = torch.max(main_cls, dim=2, keepdim=True)[0]  # (bn, num_anchor, 1)
    scores_over_thresh = (scores > threshold)[:, :, 0]  # (bn, num_anchor)
    out = []
    n = x.shape[0] // 2
    for i in range(n):
        if scores_over_thresh.sum() == 0:
            out.append({
                'rois': np.array(()),
                'rois_sub': np.array(()),
                'rois_obj': np.array(()),
                'sp_vector': np.array(()),
                'act_class_ids': np.array(()),
                'act_scores': np.array(()),
                # 'obj_scores': np.array(()),
            })
            continue

        # main_cls_per = main_cls[i, scores_over_thresh[i, :], ...].permute(1, 0)  # (num_cls, num_bbox)
        # # other_cls_per = other_cls[i, scores_over_thresh[i, :], ...].permute(1, 0)  # (num_cls, num_bbox)
        #
        # transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...] # (num_bbox, 4)
        # transformed_anchors_sub_per = transformed_anchors_sub[i, scores_over_thresh[i, :], ...] # (num_bbox, 4)
        # transformed_anchors_obj_per = transformed_anchors_obj[i, scores_over_thresh[i, :], ...] # (num_bbox, 4)
        #
        # scores_per = scores[i, scores_over_thresh[i, :], ...]

        main_cls_per = torch.cat([main_cls[i, scores_over_thresh[i, :], ...].permute(1, 0),
                                   main_cls[i+n, scores_over_thresh[i+n, :], ...].permute(1, 0)], 1)  # (num_cls, num_bbox)

        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...] # (num_bbox, 4)
        transformed_anchors_per_flip = transformed_anchors[i+n, scores_over_thresh[i+n, :], ...].clone()

        cols = x.shape[3]
        w = transformed_anchors_per_flip[:, 2] - transformed_anchors_per_flip[:, 0]
        transformed_anchors_per_flip[:, 2] = cols - transformed_anchors_per_flip[:, 0]
        transformed_anchors_per_flip[:, 0] = transformed_anchors_per_flip[:, 2] - w
        transformed_anchors_per = torch.cat([transformed_anchors_per, transformed_anchors_per_flip], 0)


        transformed_anchors_sub_per = transformed_anchors_sub[i, scores_over_thresh[i, :], ...] # (num_bbox, 4)
        transformed_anchors_sub_per_flip = transformed_anchors_sub[i+n, scores_over_thresh[i+n, :], ...].clone()

        w = transformed_anchors_sub_per_flip[:, 2] - transformed_anchors_sub_per_flip[:, 0]
        transformed_anchors_sub_per_flip[:, 2] = cols - transformed_anchors_sub_per_flip[:, 0]
        transformed_anchors_sub_per_flip[:, 0] = transformed_anchors_sub_per_flip[:, 2] - w
        transformed_anchors_sub_per = torch.cat([transformed_anchors_sub_per, transformed_anchors_sub_per_flip], 0)


        transformed_anchors_obj_per = transformed_anchors_obj[i, scores_over_thresh[i, :], ...] # (num_bbox, 4)
        transformed_anchors_obj_per_flip = transformed_anchors_obj[i+n, scores_over_thresh[i+n, :], ...].clone()

        w = transformed_anchors_obj_per_flip[:, 2] - transformed_anchors_obj_per_flip[:, 0]
        transformed_anchors_obj_per_flip[:, 2] = cols - transformed_anchors_obj_per_flip[:, 0]
        transformed_anchors_obj_per_flip[:, 0] = transformed_anchors_obj_per_flip[:, 2] - w
        transformed_anchors_obj_per = torch.cat([transformed_anchors_obj_per, transformed_anchors_obj_per_flip], 0)

        scores_per = torch.cat([scores[i, scores_over_thresh[i, :], ...], scores[i+n, scores_over_thresh[i+n, :], ...]], 0)

        if main_cls_per.shape[1] > 0:
            main_scores_ = main_cls_per[:, :]  # (num_cls, num_nms_bbox)
            # other_scores_ = other_cls_per[:, :]
            boxes_ = transformed_anchors_per[:, :]  # (num_nms_bbox, 4)
            boxes_sub_ = transformed_anchors_sub_per[:, :]  # (num_nms_bbox, 4)
            boxes_obj_ = transformed_anchors_obj_per[:, :]  # (num_nms_bbox, 4)
            sp_vector_x = (boxes_obj_[:, 0] + boxes_obj_[:, 2]) / 2 - (boxes_sub_[:, 0] + boxes_sub_[:, 2]) / 2
            sp_vector_y = (boxes_obj_[:, 1] + boxes_obj_[:, 3]) / 2 - (boxes_sub_[:, 1] + boxes_sub_[:, 3]) / 2

            sp_vector_x = sp_vector_x.reshape(-1, 1)
            sp_vector_y = sp_vector_y.reshape(-1, 1)

            sp_vector = torch.cat([sp_vector_x, sp_vector_y], 1)

            act_scores_ = main_scores_.permute(1, 0)  # (num_nms_bbox, num_cls)
            act_classes_ = main_scores_.max(dim=0)[1]
            # obj_scores_ = other_scores_.permute(1, 0)  #

            out.append({
                'rois': boxes_.cpu().numpy(),
                'rois_sub': boxes_sub_.cpu().numpy(),
                'rois_obj': boxes_obj_.cpu().numpy(),
                'sp_vector': sp_vector.cpu().numpy(),
                'act_class_ids': act_classes_.cpu().numpy(),
                'act_scores': act_scores_.cpu().numpy(),
                # 'obj_scores': obj_scores_.cpu().numpy()
            })
        else:
            out.append({
                'rois': np.array(()),
                'rois_sub': np.array(()),
                'rois_obj': np.array(()),
                'sp_vector': np.array(()),
                'act_class_ids': np.array(()),
                'act_scores': np.array(()),
                # 'obj_scores': np.array(())
            })

    return out
