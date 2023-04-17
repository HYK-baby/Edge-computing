import numpy as np
from yolo_inference_base import YOLOv3base
import tflite_runtime.interpreter as tflite


class YOLOv3tflite(YOLOv3base):
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
        该方法由子类实现
        """
        # print(inputs)
        self.interpreter.set_tensor(self.inTensorIndex, inputs)
        # inference
        self.interpreter.invoke()
        output_data = []
        for i in range(len(self.outTensorIndex)):
            output_data.append(self.interpreter.get_tensor(self.outTensorIndex[i]))
        return output_data


# def detect_img(self, org_image):
#     """
#     """
#     image_data = np.array(org_image)

#     # convert to float32 [0, 1.]
#     image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
#     image_sized = tf.image.resize_with_pad(image_data, self.input_shape[0], self.input_shape[1])
#     image_sized = tf.expand_dims(image_sized, axis=0)
#     # normalize
#     # means = tf.convert_to_tensor(self.imageStats['mean'])
#     # stds = tf.convert_to_tensor(self.imageStats['std'])
#     # image_sized = per_channel_standardization(image_sized, means, stds)
#     # model inference
#     prediction = self.detector.model.predict_on_batch(image_sized)
#     # output of object class_score and bboxes
#     pred_boxes = self.generate(prediction, self.downscale)

#     org_img_shape = tf.convert_to_tensor([org_image.height, org_image.width], dtype=tf.float32)
#     scale = tf.reduce_min(self.input_shape / org_img_shape)
#     # scale = tf.cast(scale, dtype=tf.float32)
#     aspect_ratio = tf.cast(org_img_shape, dtype=tf.float32) * scale
#     detal = (tf.cast(self.input_shape, dtype=tf.float32) - aspect_ratio)/2
#     detal = tf.reverse(detal, [0])  # tensorflow image row first, detal=[dy,dx] ==> [dx,dy]
#     print('detal:', detal)
#     # draw result
#     draw = ImageDraw.Draw(org_image)
#     # font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=np.floor(3e-2 * org_image.width + 0.5).astype('int32'))
#     thickness = 1  # (org_image.width + org_image.height) // 300
#     # thickness = (320 + 320) // 300
#     for (key, value) in pred_boxes.items():
#         bboxes_x = (value[..., 1:5:2] * self.input_shape[1] - detal[0]) / scale
#         bboxes_y = (value[..., 2:5:2] * self.input_shape[0] - detal[1]) / scale
#         bboxes_x = tf.cast(bboxes_x+.5, tf.int32)
#         bboxes_y = tf.cast(bboxes_y+.5, tf.int32)
#         print('class:{}, count:{}'.format(key, value.shape[0]))
#         print('score:{}'.format(value[..., 0]))
#         for i in range(value.shape[0]):
#             label = '{} {:.2f}'.format(self.class_names[key], value[i, 0])
#             print(label)
#             strPrediction = '{}:{},{},{},{}\n'.format(self.class_names[key], bboxes_x[i, 0], bboxes_y[i, 0], bboxes_x[i, 1], bboxes_y[i, 1])
#             print(strPrediction)
#             for o in range(thickness):
#                 draw.rectangle([bboxes_x[i, 0] + o, bboxes_y[i, 0] + o, bboxes_x[i, 1] - o, bboxes_y[i, 1] - o], outline=self.colors[key])

#         # label_size = draw.textsize(label, font)
    
#     return org_image
