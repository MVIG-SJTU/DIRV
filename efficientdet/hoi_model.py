import torch.nn as nn
import torch
from torchvision.ops.boxes import nms as nms_torch

from efficientnet import EfficientNet as EffNet
from efficientnet.utils import MemoryEfficientSwish, Swish
from efficientnet.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding

from efficientdet.model import Regressor, Classifier, SeparableConvBlock


class Union_Branch(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers, num_union_classes, num_obj_classes):
        super(Union_Branch, self).__init__()
        self.num_layers = num_layers
        self.num_anchors = num_anchors
        self.in_channels = in_channels

        self.num_union_classes = num_union_classes
        self.num_obj_classes = num_obj_classes

        # self.target_obj_classifier = Classifier(in_channels=self.in_channels, num_anchors=self.num_anchors,
        #                                         num_classes=self.num_obj_classes, num_layers=self.num_layers)

        self.action_classifier = Classifier(in_channels=self.in_channels, num_anchors=self.num_anchors,
                                            num_classes=self.num_union_classes, num_layers=self.num_layers)

        self.union_sub_regressor = Regressor(in_channels=self.in_channels, num_anchors=self.num_anchors, num_layers=self.num_layers)
        # self.union_obj_regressor = Target_Regressor(in_channels=self.in_channels, num_anchors=self.num_anchors,
        #                                             num_layers=self.num_layers, num_union_classes=self.num_union_classes)
        self.union_obj_regressor = Regressor(in_channels=self.in_channels, num_anchors=self.num_anchors, num_layers=self.num_layers)


    def forward(self, inputs):
        # union_obj_cls = self.target_obj_classifier(inputs)
        union_act_cls = self.action_classifier(inputs)
        # union_bbox_reg = self.union_box_regressor(inputs)
        union_sub_reg = self.union_sub_regressor(inputs)
        union_obj_reg = self.union_obj_regressor(inputs)

        # return union_obj_cls, union_act_cls, union_bbox_reg

        return union_act_cls, union_sub_reg, union_obj_reg


class Instance_Branch(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers, num_inst_classes, num_obj_classes):
        super(Instance_Branch, self).__init__()
        self.num_layers = num_layers
        self.num_anchors = num_anchors
        self.in_channels = in_channels

        self.num_inst_classes = num_inst_classes
        self.num_obj_classes = num_obj_classes

        self.action_classifier = Classifier(in_channels=self.in_channels, num_anchors=self.num_anchors,
                                            num_classes=self.num_inst_classes, num_layers=self.num_layers)

        self.object_classifier = Classifier(in_channels=self.in_channels, num_anchors=self.num_anchors,
                                            num_classes=self.num_obj_classes, num_layers=self.num_layers)

        self.object_regressor = Regressor(in_channels=self.in_channels, num_anchors=self.num_anchors, num_layers=self.num_layers)

    def forward(self, inputs):
        inst_act_cls = self.action_classifier(inputs)
        inst_obj_cls = self.object_classifier(inputs)
        inst_bbox_reg = self.object_regressor(inputs)

        return inst_act_cls, inst_obj_cls, inst_bbox_reg


# class Target_Regressor(nn.Module):
#     """
#     modified by Zylo117
#     """
#
#     def __init__(self, in_channels, num_anchors, num_layers, num_union_classes, onnx_export=False):
#         super(Target_Regressor, self).__init__()
#         self.num_layers = num_layers
#
#         self.num_union_classes = num_union_classes
#         self.conv_list = nn.ModuleList(
#             [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
#         self.bn_list = nn.ModuleList(
#             [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
#              range(5)])
#         self.header = SeparableConvBlock(in_channels, num_anchors * num_union_classes * 4, norm=False, activation=False)
#         self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
#
#     def forward(self, inputs):
#         feats = []
#         for feat, bn_list in zip(inputs, self.bn_list):
#             for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
#                 feat = conv(feat)
#                 feat = bn(feat)
#                 feat = self.swish(feat)
#             feat = self.header(feat)
#
#             feat = feat.permute(0, 2, 3, 1)  # (batch_size, height, width, num_anchor*num_union_class*4)
#             feat = feat.contiguous().view(feat.shape[0], -1, 4)  # (batch_size, h*w*num_anchor, 4)
#             # feat = feat.contiguous().view(feat.shape[0], -1, self.num_union_classes, 4)  # (batch_size, h*w*num_anchor, num_union_class*4)
#
#             feats.append(feat)
#
#         feats = torch.cat(feats, dim=1)  # (batch_size, h*w*feat_num*num_anchor, 4)
#
#         return feats






