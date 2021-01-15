import numpy as np

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

sub_union_map = np.zeros(len(label_to_class), dtype=np.uint8)
for uid in label_to_class:
    for sid in sub_label_to_class:
        if sub_label_to_class[sid] == label_to_class[uid][0]:
            sub_union_map[uid] = sid
            break


def to_onehot(label, label_num):
    if isinstance(label, int) or isinstance(id, np.int32) or isinstance(id, np.int64):
        tmp = np.zeros(label_num)
        tmp[label] = 1
    elif isinstance(label, list) or isinstance(id, np.ndarray):
        tmp = np.zeros(label_num)
        label = np.array(label)
        assert len(label.shape) == 1
        if label.shape[0] > 0:
            tmp[label] = 1
    else:
        raise (Exception, "Only int or list is allowed to transform to one hot")
    return tmp


def single_iou(a, b, need_area = False):
    # a(x1, y1, x2, y2)
    # b(x1, y1, x2, y2)

    area = (b[2] - b[0]) * (b[3] - b[1])
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    iw = max(iw, 0)
    ih = max(ih, 0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + area - iw * ih
    ua = max(ua, 1e-8)

    intersection = iw * ih
    IoU = intersection / ua
    if need_area:
        return IoU, intersection, ua
    else:
        return IoU


def single_ioa(a, b, need_area = False):
    # a(x1, y1, x2, y2)
    # b(x1, y1, x2, y2)

    area = (b[2] - b[0]) * (b[3] - b[1])
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])

    iw = max(iw, 0)
    ih = max(ih, 0)

    area = max(area, 1e-8)
    intersection = iw * ih
    IoA = intersection / area

    if need_area:
        return IoA, intersection, area
    else:
        return IoA


def single_inter(a, b):
    inter = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    if inter[0] > inter[2] or inter[1] > inter[3]:
        inter = [0.0, 0.0, 0.0, 0.0]
    return np.array(inter)


def single_union(a, b):
    inter = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
    if inter[0] > inter[2] or inter[1] > inter[3]:
        inter = [0.0, 0.0, 0.0, 0.0]
    return np.array(inter)


def transform_action(inst_score, mode):
    assert mode in {"subject", "object"}

    num_union_act = len(label_to_class)
    num_sub_act = len(sub_label_to_class)
    num_obj_act = len(obj_label_to_class)

    res = np.zeros(num_union_act)
    ids = np.arange(num_union_act)

    if mode == "object":
        res = inst_score[num_sub_act:]
        return res
    else:
        res[ids] = inst_score[sub_union_map[ids]]
        return res