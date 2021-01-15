def apply_prior(scores, obj_cls):
    assert len(scores) == 25
    if obj_cls != 35:  # not a snowboard, then the action is impossible to be snowboard
        scores[24] = 0

    if obj_cls != 83:  # not a book, then the action is impossible to be read
        scores[23] = 0

    if obj_cls != 36:  # not a sports ball, then the action is impossible to be kick
        scores[21] = 0

    if (obj_cls != 45) and (obj_cls != 43) and (obj_cls != 46) and (obj_cls != 50):  # not 'wine glass', 'bottle', 'cup', 'bowl', then the action is impossible to be drink
        scores[20] = 0

    if obj_cls != 40:  # not a skateboard, then the action is impossible to be skateboard
        scores[19] = 0

    if obj_cls != 41:  # not a surfboard, then the action is impossible to be surfboard
        scores[18] = 0

    if obj_cls != 34:  # not a ski, then the action is impossible to be ski
        scores[17] = 0

    if obj_cls != 72:  # not a laptop, then the action is impossible to be work on computer
        scores[16] = 0

    if (obj_cls != 86) and (obj_cls != 47) and (obj_cls != 48):  # not 'scissors', 'fork', 'knife', then the action is impossible to be cur instr
        scores[14] = 0

    if (obj_cls != 36) and (obj_cls != 33): # not 'sports ball', 'frisbee', then the action is impossible to be throw and catch
        scores[12] = 0
        scores[13] = 0

    if obj_cls != 76:  # not a cellphone, then the action is impossible to be talk_on_phone
        scores[10] = 0

    if (obj_cls != 14) and (obj_cls != 66) and (obj_cls != 69) and (obj_cls != 64) and (obj_cls != 62) and (obj_cls != 61):  # not 'bench', 'dining table', 'toilet', 'bed', 'couch', 'chair', then the action is impossible to be lay
        scores[9] = 0

    if (obj_cls != 35) and (obj_cls != 34) and (obj_cls != 40) and (obj_cls != 41):  # not 'snowboard', 'skis', 'skateboard', 'surfboard', then the action is impossible to be jump
        scores[8] = 0

    if (obj_cls != 51) and (obj_cls != 52) and (obj_cls != 53) and (obj_cls != 54) and (obj_cls != 55) and (obj_cls != 56) and (obj_cls != 57) and (obj_cls != 58) and (obj_cls != 59) and (obj_cls != 60):  # not ''banana', 'apple', 'sandwich', 'orange', 'carrot', 'broccoli', 'hot dog', 'pizza', 'cake', 'donut', then the action is impossible to be eat_obj
        scores[6] = 0

    if (obj_cls != 47) and (obj_cls != 48) and (obj_cls != 49):  # not 'fork', 'knife', 'spoon', then the action is impossible to be eat_instr
        scores[7] = 0

    if (obj_cls != 42) and (obj_cls != 38): # not 'tennis racket', 'baseball bat', then the action is impossible to be hit_instr
        scores[4] = 0

    if (obj_cls != 36):  # not 'sports ball, then the action is impossible to be hit_obj
        scores[5] = 0

    if (obj_cls != 1) and (obj_cls != 3) and (obj_cls != 5) and (obj_cls != 7) and (obj_cls != 8) and (obj_cls != 6) and (obj_cls != 4) and (obj_cls != 2) and (obj_cls != 18) and (obj_cls != 21): # not 'bicycle', 'motorcycle', 'bus', 'truck', 'boat', 'train', 'airplane', 'car', 'horse', 'elephant', then the action is impossible to be ride
        scores[2] = 0

    if (obj_cls != 1) and (obj_cls != 3) and (obj_cls != 18) and (obj_cls != 21) and (obj_cls != 14) and (obj_cls != 61) and (obj_cls != 62) and (obj_cls != 64) and (obj_cls != 69) and (obj_cls != 66) and (obj_cls != 32) and (obj_cls != 30) and (obj_cls != 26): # not 'bicycle', 'motorcycle', 'horse', 'elephant', 'bench', 'chair', 'couch', 'bed', 'toilet', 'dining table', 'suitcase', 'handbag', 'backpack', then the action is impossible to be sit
        scores[1] = 0

    if (obj_cls == 0):  # "person",  then the action is impossible to be cut_obj
        scores[15] = 0

    return scores
