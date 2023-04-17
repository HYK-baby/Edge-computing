import numpy as np
from yolo_inference_base import YOLOv3base, getGrid
import tflite_runtime.interpreter as tflite


class Mydetector(YOLOv3base):
    def __init__(self,
                 model_path='mydensenet_det.tflite',
                 input_shape=[320, 320],
                 num_classes=20,
                 anchors=[[[81, 82], [135, 169], [344, 319]], [[10, 14], [23, 27], [37, 58]]],
                 score_threshold=.5,
                 iou_threshold=.6
                 ):
        super().__init__(model_path, input_shape, num_classes, anchors, score_threshold, iou_threshold)
        self.createModel()

    def createModel(self):
        print('================tflite model loading.')
        self.interpreter = tflite.Interpreter(self.model_path, num_threads=4)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        print(input_details)
        
        self.inTensorIndex = input_details[0]['index']
        self.outTensorIndex = []
        for i in range(len(output_details)):
            self.outTensorIndex.append(output_details[i]['index'])
        
        # allocate_tensors
        self.interpreter.allocate_tensors()

    def preprocess(self, inputs):
        inputs = np.expand_dims(inputs, axis=0)/255.0
        return inputs.astype(np.float32)
        # return np.expand_dims(inputs, axis=0)

    def modelrunning(self, inputs):
        """
        run detector
        """
        # print(inputs)
        self.interpreter.set_tensor(self.inTensorIndex, inputs)
        # inference
        self.interpreter.invoke()
        output_data = []
        for i in range(len(self.outTensorIndex)):
            output_data.append(self.interpreter.get_tensor(self.outTensorIndex[i]))
        return output_data
    
    def proprocess(self, model_out):
        """
        anchor free proprocess output bboxes.
        model_out: list of prediction shape = (b, h, w, C+5)
        """
        box_xy, box_wh = [], []
        box_confidence = []
        box_class_probs = []
        levels = len(model_out)
        for i in range(levels):
            grid_shape = model_out[i].shape[1:3]
            grid_shape = grid_shape[::-1]
            feat_w = grid_shape[0]
            feat_h = grid_shape[1]
            # --------------------------------------------------- #
            feats = model_out[i]  # (b, h, w, C+5)
            # ------------------------------------------#
            #   获得预测框的置信度与类别
            # ------------------------------------------#
            sub_box_confidence = self._sigmoid(feats[..., 0:1])
            valid_index = np.where(sub_box_confidence[..., 0] >= self.score_threshold)
            # print('valid_index:{}'.format(valid_index))
            feats = feats[..., 1:]
            valid_feats = feats[valid_index]  # (n, 4+num_classes)

            grid = getGrid(feat_w, feat_h)     # (h, w, 2)
            grid = np.reshape(grid, [1, feat_h, feat_w, 2])
            grid = grid[valid_index]
            # print(grid)
            sub_box_xy = (valid_feats[..., 0:2] + grid) / grid_shape
            sub_box_wh = np.exp(valid_feats[..., 2:4]) / grid_shape
            
            sub_box_class_probs = self._sigmoid(valid_feats[..., 4:])
            sub_box_confidence = sub_box_confidence[valid_index]

            # box_xy.append(np.reshape(sub_box_xy, [-1, feat_h*feat_w, 2]))
            # box_wh.append(np.reshape(sub_box_wh, [-1, feat_h*feat_w, 2]))
            # box_confidence.append(np.reshape(sub_box_confidence, [-1, feat_h*feat_w, 1]))
            # box_class_probs.append(np.reshape(sub_box_class_probs, [-1, feat_h*feat_w, self.num_classes]))
            box_xy.append(sub_box_xy)
            box_wh.append(sub_box_wh)
            box_confidence.append(sub_box_confidence)
            box_class_probs.append(sub_box_class_probs)
        box_xy = np.concatenate(box_xy, axis=0)
        box_wh = np.concatenate(box_wh, axis=0)
        box_confidence = np.concatenate(box_confidence, axis=0)
        box_class_probs = np.concatenate(box_class_probs, axis=0)
        # concat
        pred_bboxes = np.concatenate([box_confidence, box_xy, box_wh, box_class_probs], axis=-1)
        return pred_bboxes
