import numpy as np


def getGrid(w, h, step=1):
    shift_1 = np.arange(0, w) * step
    shift_2 = np.arange(0, h) * step
    mesh = np.meshgrid(shift_1, shift_2)
    X = np.reshape(mesh[0], [h, w, 1])
    Y = np.reshape(mesh[1], [h, w, 1])
    return np.concatenate([X, Y], axis=-1)


def _iou(b1, b2):
    ix1 = np.maximum(b1[0], b2[0])
    iy1 = np.maximum(b1[1], b2[1])
    ix2 = np.minimum(b1[2], b2[2])
    iy2 = np.minimum(b1[3], b2[3])
    intersection = (ix2-ix1) * (iy2-iy1)
    if intersection < 0:
        return 0.
    union = (b1[2]-b1[0]) * (b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - intersection
    return intersection / union


def non_max_suppression(boxes, score, iou_threshold):
    """
    return selected box's index. shape: (N, )
    """
    N = boxes.shape[0]
    # if N < 1:
    #    return 0
    mask = np.ones(N, dtype=np.int8)
    ordering = np.argsort(score)
    for i in range(N):
        if mask[ordering[i]] == 0:
            continue
        a = boxes[ordering[i]]
        for j in range(i+1, N):
            if _iou(a, boxes[ordering[j]]) >= iou_threshold:
                mask[ordering[j]] = 0
    mask = mask.astype(bool)
    return ordering[mask]


class YOLOv3base(object):
    def __init__(self,
                 model_path='model_data/ep150-loss9.545-val_loss12.151.h5',
                 input_shape=[320, 320],
                 num_classes=20,
                 anchors=[[[81, 82], [135, 169], [344, 319]], [[10, 14], [23, 27], [37, 58]]],
                 score_threshold=.5,
                 iou_threshold=.6
                 ):
        self.model_path = model_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchors = anchors
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def preprocess(self, inputs):
        return inputs / 255

    def modelrunning(self, inputs):
        """
        该方法由子类实现
        """
        outputs = [np.zeros([1, 10, 10, 3 * (self.num_classes+5)])]
        outputs.append(np.zeros([1, 20, 20, 3 * (self.num_classes+5)]))
        return outputs

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def proprocess(self, model_out):
        """
        yolov3 prediction proprocess output bboxes.
        model_out: list of prediction shape = (b, h, w, A, C+5)
        """
        box_xy, box_wh = [], []
        box_confidence = []
        box_class_probs = []
        levels = len(model_out)
        for i in range(levels):
            num_anchors = len(self.anchors[i])
            anchors = np.reshape(self.anchors[i], [1, 1, 1, num_anchors, 2])
            grid_shape = model_out[i].shape[1:3]
            feat_h = grid_shape[0]
            feat_w = grid_shape[1]
            # --------------------------------------------------- #
            feats = np.reshape(model_out[i], [-1, feat_h, feat_w, num_anchors, self.num_classes + 5])

            grid = getGrid(feat_w, feat_h)     # (h, w, 2)
            grid = np.reshape(grid, [1, feat_h, feat_w, 1, 2])
            sub_box_xy = (self._sigmoid(feats[..., :2]) + grid) / grid_shape
            sub_box_wh = np.exp(feats[..., 2:4]) * anchors / self.input_shape[::-1]
            # ------------------------------------------#
            #   获得预测框的置信度与类别
            # ------------------------------------------#
            sub_box_confidence = self._sigmoid(feats[..., 4:5])
            sub_box_class_probs = self._sigmoid(feats[..., 5:])

            box_xy.append(np.reshape(sub_box_xy, [-1, feat_h*feat_w*num_anchors, 2]))
            box_wh.append(np.reshape(sub_box_wh, [-1, feat_h*feat_w*num_anchors, 2]))
            box_confidence.append(np.reshape(sub_box_confidence, [-1, feat_h*feat_w*num_anchors, 1]))
            box_class_probs.append(np.reshape(sub_box_class_probs, [-1, feat_h*feat_w*num_anchors, self.num_classes]))
        box_xy = np.concatenate(box_xy, axis=1)
        box_wh = np.concatenate(box_wh, axis=1)
        box_confidence = np.concatenate(box_confidence, axis=1)
        box_class_probs = np.concatenate(box_class_probs, axis=1)
        # concat
        pred_bboxes = np.concatenate([box_confidence, box_xy, box_wh, box_class_probs], axis=-1)
        # select boxes
        valid_index = np.where(pred_bboxes[..., 0] >= self.score_threshold)
        # print('valid_index:{}'.format(valid_index))
        valid_bboxes = pred_bboxes[valid_index]  # (n, 1+4+num_classes)
        return valid_bboxes

    def filterbboxes(self, pred_bboxes):
        """
        select boxes by confidence >= score_threshold.
        pred_bboxes: (M, c+5)
        """
        # valid_index = np.where(pred_bboxes[..., 0] >= self.score_threshold)
        # print('valid_index:{}'.format(valid_index))
        # valid_bboxes = pred_bboxes[valid_index]  # (n, 1+4+num_classes)
        valid_bboxes = pred_bboxes

        all_cls_score = valid_bboxes[..., 0:1] * valid_bboxes[..., 5:]
        # valid left top right bottom
        box_lt = valid_bboxes[..., 1:3] - valid_bboxes[..., 3:5]/2
        box_rb = box_lt + valid_bboxes[..., 3:5]
        box_lt = np.where(box_lt < 0, .0, box_lt)
        box_rb = np.where(box_rb > 1, 1.0, box_rb)
        all_boxes = np.concatenate([box_lt, box_rb], axis=-1)
        # print('all_cls_score:', all_cls_score)
        # print('all_boxes:', all_boxes)
        # NMS
        lstObject = []
        for i in range(self.num_classes):
            c = np.where(all_cls_score[:, i] >= self.score_threshold)
            to_nms_boxes = all_boxes[c]
            to_nms_score = all_cls_score[c][:, i]
            selected = non_max_suppression(to_nms_boxes, to_nms_score, self.iou_threshold)
            for j in range(selected.shape[0]):
                obj = dict()
                obj['box'] = to_nms_boxes[selected[j]]
                obj['score'] = to_nms_score[selected[j]]
                obj['class'] = i
                lstObject.append(obj)
        return lstObject

    def predicted(self, image):
        """
        np
        """
        image = self.preprocess(image)
        model_out = self.modelrunning(image)
        pred_bboxes = self.proprocess(model_out)
        return self.filterbboxes(pred_bboxes)
