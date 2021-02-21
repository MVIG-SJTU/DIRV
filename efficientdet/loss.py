import torch
import torch.nn as nn
import cv2
import numpy as np
import sys
import json
import copy

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import postprocess, invert_affine, display

np.set_printoptions(precision=3, suppress=True, threshold=np.inf)

def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


def calc_ioa(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    intersection = iw * ih
    area = torch.clamp(area, min=1e-8)
    IoA = intersection / area
    IoA[torch.isnan(IoA)] = 1

    return IoA


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, **kwargs):

        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]  # (h*w*feat_num, num_classes)
            regression = regressions[j, :, :]  # (h*w*feat_num, num_anchor*4)

            bbox_annotation = annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]  # (num_boxes, 5)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    classification_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(torch.tensor(0).to(dtype))

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # 不同stride

            # compute the loss for classification
            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0  # IoU < 0.4

            positive_indices = torch.ge(IoU_max, 0.5)  # IoU > 0.5

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1  # set the corresponding categories as 1

            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce # 分类loss

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)  # ignore loss if IoU is too small

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        # debug
        imgs = kwargs.get('imgs', None)
        if imgs is not None:
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            obj_list = kwargs.get('obj_list', None)
            out = postprocess(imgs.detach(),
                              torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
                              regressBoxes, clipBoxes,
                              0.5, 0.3)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            display(out, imgs, obj_list, imshow=False, imwrite=True)

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True)


class Instance_Loss(nn.Module):
    def __init__(self, dataset="vcoco"):
        super(Instance_Loss,self).__init__()
        self.focal_loss = FocalLoss()
        self.bce_loss = nn.BCELoss()
        self.dataset = dataset
        if dataset == "hico-det":
            with open("datasets/hico_20160224_det/hico_processed/hico-det_verb_count.json", "r") as file:
                verb_count = json.load(file)

            verb_count = verb_count + verb_count
            verb_count = np.array(verb_count)
            verb_count = np.log(verb_count)
            verb_weight = np.mean(verb_count) / verb_count
            verb_weight = np.round(verb_weight, 1)
            self.verb_weight = torch.from_numpy(verb_weight)

    def forward(self, act_classifications, obj_classifications, obj_regressions, anchors, inst_annotations,  **kwargs):
        anchors = anchors.float()
        act_classifications = act_classifications.float()

        alpha = 0.25
        gamma = 2.0
        batch_size = act_classifications.shape[0]
        act_classification_losses = []
        obj_classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):
            act_classification = act_classifications[j, :, :]  # (h*w*feat_num, num_act_classes)
            obj_classification = obj_classifications[j, :, :]  # (h*w*feat_num, num_obj_classes)
            regression = obj_regressions[j, :, :]  # (h*w*feat_num, num_anchor*4)

            bbox_annotation = inst_annotations[j, :, :5]
            act_annotation_oh = inst_annotations[j, :, 5:]

            act_annotation_oh = act_annotation_oh[bbox_annotation[:, 4] != -1]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]  # (num_boxes, 5)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    act_classification_losses.append(torch.tensor(0).to(dtype).cuda())
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    obj_classification_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    act_classification_losses.append(torch.tensor(0).to(dtype))
                    regression_losses.append(torch.tensor(0).to(dtype))
                    obj_classification_losses.append(torch.tensor(0).to(dtype))
                continue

            obj_classification = torch.clamp(obj_classification, 1e-4, 1.0 - 1e-4)
            act_classification = torch.clamp(act_classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            act_targets = torch.ones_like(act_classification) * -1
            obj_targets = torch.ones_like(obj_classification) * -1
            if torch.cuda.is_available():
                act_targets = act_targets.cuda()
                obj_targets = obj_targets.cuda()

            obj_targets[torch.lt(IoU_max, 0.4), :] = 0  # IoU < 0.4
            act_targets[torch.lt(IoU_max, 0.4), :] = 0  # IoU < 0.4

            positive_indices = torch.ge(IoU_max, 0.5)  # IoU > 0.5

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]
            assigned_act_annotation = act_annotation_oh[IoU_argmax, :]

            act_targets[positive_indices, :] = 0
            obj_targets[positive_indices, :] = 0

            obj_targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1  # set the corresponding categories as 1
            act_targets[positive_indices, :] = assigned_act_annotation[positive_indices, :]  # set the corresponding categories as 1

            foreground = torch.max(act_targets, dim=1)[0] > 0
            act_targets = act_targets[foreground]
            act_classification = act_classification[foreground]

            alpha_factor_obj = torch.ones_like(obj_targets) * alpha

            if torch.cuda.is_available():
                alpha_factor_obj = alpha_factor_obj.cuda()

            alpha_factor_obj = torch.where(torch.eq(obj_targets, 1.), alpha_factor_obj, 1. - alpha_factor_obj)

            obj_focal_weight = torch.where(torch.eq(obj_targets, 1.), 1. - obj_classification, obj_classification)
            obj_focal_weight = alpha_factor_obj * torch.pow(obj_focal_weight, gamma)

            obj_bce = -(obj_targets * torch.log(obj_classification) + (1.0 - obj_targets) * torch.log(1.0 - obj_classification))
            act_bce = -(act_targets * torch.log(act_classification) + (1.0 - act_targets) * torch.log(1.0 - act_classification))

            obj_cls_loss = obj_focal_weight * obj_bce  # classification loss

            if self.dataset == "vcoco":
                act_cls_loss = act_bce
            else:
                act_cls_loss = act_bce * self.verb_weight.to(dtype).cuda( )

            obj_zeros = torch.zeros_like(obj_cls_loss)
            act_zeros = torch.zeros_like(act_cls_loss)
            if torch.cuda.is_available():
                obj_zeros = obj_zeros.cuda()
                act_zeros = act_zeros.cuda()
            obj_cls_loss = torch.where(torch.ne(obj_targets, -1.0), obj_cls_loss, obj_zeros)  # ignore loss if IoU is too small
            act_cls_loss = torch.where(torch.ne(act_targets, -1.0), act_cls_loss, act_zeros)  # ignore loss if IoU is too small

            obj_classification_losses.append(obj_cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))
            act_classification_losses.append(act_cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        # debug
        imgs = kwargs.get('imgs', None)
        if imgs is not None:
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            obj_list = kwargs.get('obj_list', None)
            out = postprocess(imgs.detach(),
                              torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), obj_classifications.detach(),
                              regressBoxes, clipBoxes,
                              0.5, 0.3)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            display(out, imgs, obj_list, imshow=False, imwrite=True)

        return torch.stack(act_classification_losses).mean(dim=0, keepdim=True), \
                torch.stack(obj_classification_losses).mean(dim=0, keepdim=True), \
                torch.stack(regression_losses).mean(dim=0, keepdim=True)


class Union_Loss(nn.Module):
    def __init__(self, dataset="vcoco"):
        super(Union_Loss, self).__init__()
        self.dataset = dataset
        if self.dataset == "hico-det":
            with open("datasets/hico_20160224_det/hico_processed/hico-det_verb_count.json", "r") as file:
                hoi_count = json.load(file)

            hoi_count = np.log(hoi_count)
            hoi_weight = np.mean(hoi_count) / hoi_count
            hoi_weight = np.round(hoi_weight, 1)
            self.hoi_weight = torch.from_numpy(hoi_weight)

    def forward(self, act_classifications, sub_regressions, obj_regressions, anchors, union_annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = act_classifications.shape[0]
        # obj_classification_losses = []
        act_classification_losses = []
        # regression_losses = []
        sub_regression_losses = []
        obj_regression_losses = []
        diff_regression_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            act_classification = act_classifications[j, :, :] # (h*w*feat_num, num_classes)
            sub_regression = sub_regressions[j, :, :]  # (h*w*feat_num*num_anchor, 4)
            obj_regression = obj_regressions[j, :, :]  # (h*w*feat_num*num_anchor, num_union_class, 4)

            bbox_annotation = union_annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 0] >= 0] # (num_union, K)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    act_classification_losses.append(torch.tensor(0).to(dtype).cuda())
                    sub_regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    obj_regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    diff_regression_losses.append(torch.tensor(0).to(dtype).cuda())

                else:
                    act_classification_losses.append(torch.tensor(0).to(dtype))
                    sub_regression_losses.append(torch.tensor(0).to(dtype))
                    obj_regression_losses.append(torch.tensor(0).to(dtype))
                    diff_regression_losses.append(torch.tensor(0).to(dtype))

                continue

            act_classification = torch.clamp(act_classification, 1e-4, 1.0 - 1e-4) # (h*w*feat_num, num_classes)

            IoU = calc_iou(anchor[:, :], bbox_annotation[:, 8:12])  # (h*w*anchor_num, num_union)
            IoA_sub = calc_ioa(anchor[:, :], bbox_annotation[:, :4]) # (h*w*anchor_num, num_union)
            IoA_obj = calc_ioa(anchor[:, :], bbox_annotation[:, 4:8]) # (h*w*anchor_num, num_union)

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # 不同stride, (h*w*anchor_num, )

            Union_IoU = (IoU > 0.25) * (IoA_sub > 0.25) * (IoA_obj > 0.25)
            Union_IoU = Union_IoU.float()

            IoU_max_ge, IoU_argmax_ge = torch.max(0.5 * (IoU+torch.sqrt(IoA_sub*IoA_obj))*Union_IoU, dim=1)  # (h*w*anchor_num, )

            # compute the loss for classification
            act_targets = torch.ones_like(act_classification, dtype=torch.float32) * -1 # (h*w*feat_num, num_classes)

            if torch.cuda.is_available():
                act_targets = act_targets.cuda()

            act_targets[torch.lt(IoU_max, 0.4), :] = 0  # IoU < 0.4,

            positive_indices = torch.max(Union_IoU, dim=1)[0]>0 # (h*w*anchor_num, 1)
            positive_indices_reg = torch.ge(IoU_max_ge, 0.1) # actually same as above

            num_positive_anchors = positive_indices.sum()

            assigned_act_annotation_all_fore = torch.mm(Union_IoU, bbox_annotation[:, 13:])  # (h*w*anchor_num, num_class)
            assigned_act_annotation_all_fore = torch.clamp(assigned_act_annotation_all_fore, 0, 1)  # (h*w*anchor_num, num_class)

            assigned_act_annotation = bbox_annotation[IoU_argmax_ge, 13:]  # (h*w*anchor_num, num_class)
            assigned_annotations = bbox_annotation[IoU_argmax_ge, :]

            assigned_act_annotations_ignore = assigned_act_annotation_all_fore - assigned_act_annotation
            assigned_act_annotations_ignore = assigned_act_annotations_ignore[positive_indices]
            # assert assigned_act_annotations_ignore.max() <= 1
            # assert assigned_act_annotations_ignore.min() >= 0

            act_targets[positive_indices, :] = 0
            act_targets[positive_indices, :] = assigned_act_annotation[positive_indices, :]

            act_targets = act_targets[positive_indices]
            act_classification = act_classification[positive_indices]
            act_targets = act_targets - assigned_act_annotations_ignore

            alpha_factor_act = torch.ones_like(act_targets, dtype=torch.float32) * alpha

            if torch.cuda.is_available():
                alpha_factor_act = alpha_factor_act.cuda()
            alpha_factor_act = torch.where(torch.eq(act_targets, 1.), alpha_factor_act, 1. - alpha_factor_act)

            focal_weight_act = torch.where(torch.eq(act_targets, 1.), 1. - act_classification, act_classification)
            focal_weight_act = alpha_factor_act * torch.pow(focal_weight_act, gamma)

            act_bce = -(act_targets * torch.log(act_classification) + (1.0 - act_targets) * torch.log(1.0 - act_classification))

            if self.dataset == "vcoco":
                act_cls_loss = focal_weight_act * act_bce
            else:
                act_cls_loss = focal_weight_act * act_bce * self.hoi_weight.to(dtype).cuda()  # classification loss

            act_zeros = torch.zeros_like(act_cls_loss)

            act_cls_loss = torch.where(torch.ne(act_targets, -1.0), act_cls_loss, act_zeros)  # ignore loss if IoU is too small
            act_classification_losses.append(act_cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices_reg.sum() > 0:
                assigned_annotations_sub = assigned_annotations[positive_indices_reg, 0:4]
                assigned_annotations_obj = assigned_annotations[positive_indices_reg, 4:8]

                sub_regression_pi = sub_regression[positive_indices_reg, :]
                obj_regression_pi = obj_regression[positive_indices_reg, :]

                anchor_widths_pi = anchor_widths[positive_indices_reg]
                anchor_heights_pi = anchor_heights[positive_indices_reg]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices_reg]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices_reg]

                sub_regression_loss = regression_loss(anchor_widths_pi, anchor_heights_pi, anchor_ctr_x_pi, anchor_ctr_y_pi,
                                                      assigned_annotations_sub, sub_regression_pi)
                obj_regression_loss = regression_loss(anchor_widths_pi, anchor_heights_pi, anchor_ctr_x_pi, anchor_ctr_y_pi,
                                                      assigned_annotations_obj, obj_regression_pi)

                diff_regression_loss = union_regression_loss(anchor_widths_pi, anchor_heights_pi, anchor_ctr_x_pi, anchor_ctr_y_pi,
                                                             assigned_annotations_sub, assigned_annotations_obj, sub_regression_pi,
                                                             obj_regression_pi)

                sub_regression_losses.append(sub_regression_loss.mean())
                obj_regression_losses.append(obj_regression_loss.mean())
                diff_regression_losses.append(diff_regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    sub_regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    obj_regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    diff_regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    sub_regression_losses.append(torch.tensor(0).to(dtype))
                    obj_regression_losses.append(torch.tensor(0).to(dtype))
                    diff_regression_losses.append(torch.tensor(0).to(dtype))

        return torch.stack(act_classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(sub_regression_losses).mean(dim=0, keepdim=True), \
               torch.stack(obj_regression_losses).mean(dim=0, keepdim=True), \
               torch.stack(diff_regression_losses).mean(dim=0, keepdim=True)


def regression_loss(anchor_widths_pi, anchor_heights_pi, anchor_ctr_x_pi, anchor_ctr_y_pi, assigned_annotations, regression_pi):
    gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
    gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
    gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
    gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

    # efficientdet style
    gt_widths = torch.clamp(gt_widths, min=1)
    gt_heights = torch.clamp(gt_heights, min=1)

    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
    targets_dw = torch.log(gt_widths / anchor_widths_pi)
    targets_dh = torch.log(gt_heights / anchor_heights_pi)

    targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
    targets = targets.t()

    regression_diff = torch.abs(targets - regression_pi)

    regression_loss = torch.where(
        torch.le(regression_diff, 1.0 / 9.0),
        0.5 * 9.0 * torch.pow(regression_diff, 2),
        regression_diff - 0.5 / 9.0
    )
    return regression_loss


def union_regression_loss(anchor_widths_pi, anchor_heights_pi, anchor_ctr_x_pi, anchor_ctr_y_pi,
                          assigned_annotations_sub, assigned_annotations_obj, regression_pi_sub, regression_pi_obj):
    gt_dists_x = (assigned_annotations_obj[:, 0] + assigned_annotations_obj[:, 2]) / 2 - \
                (assigned_annotations_sub[:, 0] + assigned_annotations_sub[:, 2]) / 2
    gt_dists_y = (assigned_annotations_obj[:, 1] + assigned_annotations_obj[:, 3]) / 2 - \
                (assigned_annotations_sub[:, 1] + assigned_annotations_sub[:, 3]) / 2

    gt_widths_obj = assigned_annotations_obj[:, 2] - assigned_annotations_obj[:, 0]
    gt_widths_sub = assigned_annotations_sub[:, 2] - assigned_annotations_sub[:, 0]
    gt_heights_obj = assigned_annotations_obj[:, 3] - assigned_annotations_obj[:, 1]
    gt_heights_sub = assigned_annotations_sub[:, 3] - assigned_annotations_sub[:, 1]

    # efficientdet style
    gt_widths_obj = torch.clamp(gt_widths_obj, min=1)
    gt_widths_sub = torch.clamp(gt_widths_sub, min=1)
    gt_heights_obj = torch.clamp(gt_heights_obj, min=1)
    gt_heights_sub = torch.clamp(gt_heights_sub, min=1)

    gt_widths_ratio = gt_widths_obj / gt_widths_sub
    gt_heights_ratio = gt_heights_obj / gt_heights_sub

    targets_dist_dx = gt_dists_x / anchor_widths_pi
    targets_dist_dy = gt_dists_y / anchor_heights_pi
    targets_ratio_dw = torch.log(gt_widths_ratio)
    targets_ratio_dh = torch.log(gt_heights_ratio)

    targets = torch.stack((targets_dist_dy, targets_dist_dx, targets_ratio_dh, targets_ratio_dw))
    targets = targets.t()

    regression_pi = regression_pi_obj - regression_pi_sub

    regression_diff = torch.abs(targets - regression_pi)

    regression_loss = torch.where(
        torch.le(regression_diff, 1.0 / 9.0),
        0.5 * 9.0 * torch.pow(regression_diff, 2),
        regression_diff - 0.5 / 9.0
    )
    return regression_loss
