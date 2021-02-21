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

        self.action_classifier = Classifier(in_channels=self.in_channels, num_anchors=self.num_anchors,
                                            num_classes=self.num_union_classes, num_layers=self.num_layers)

        self.union_sub_regressor = Regressor(in_channels=self.in_channels, num_anchors=self.num_anchors, num_layers=self.num_layers)
        self.union_obj_regressor = Regressor(in_channels=self.in_channels, num_anchors=self.num_anchors, num_layers=self.num_layers)


    def forward(self, inputs):
        union_act_cls = self.action_classifier(inputs)
        union_sub_reg = self.union_sub_regressor(inputs)
        union_obj_reg = self.union_obj_regressor(inputs)

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







