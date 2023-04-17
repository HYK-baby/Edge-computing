import numpy as np
import tflite_runtime.interpreter as tflite


class Keyframe():
    def __init__(self,
                 threshold,
                 model_path='myencode.tflite',
                 input_shape=[320, 320]
                 ):
        self.model_path = model_path
        self.interpreter = tflite.Interpreter(self.model_path, num_threads=2)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        print(input_details)
        self.inTensorIndex = input_details[0]['index']
        self.outTensorIndex = output_details[0]['index']
        # allocate_tensors
        self.interpreter.allocate_tensors()
        self.invoke = False
        self.features = None
        self.distance_threshold = threshold

    def judge(self, inputs):
        inputs = np.expand_dims(inputs, axis=0)/255.0
        inputs = inputs.astype(np.float32)
        # inference
        self.interpreter.set_tensor(self.inTensorIndex, inputs)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.outTensorIndex)
        isKeyframe = False
        distance = 0.
        if self.invoke:
            isKeyframe = False
            # 大于阈值,为关键帧
            distance = self._distance(output_data, self.features)
            if np.max(distance) > self.distance_threshold:
                self.features = output_data
                isKeyframe = True
        else:
            self.features = output_data
            self.invoke = True
            isKeyframe = True
        return isKeyframe, distance

    def _distance(self, fa, fb):
        type = 'l1'
        dis = 0.5
        c = fa.shape[-1]
        fa = fa.reshape((-1, c))
        fb = fb.reshape((-1, c))
        if type == 'cos':
            # cos distance
            l2_a = np.linalg.norm(fa, axis=-1)
            l2_b = np.linalg.norm(fb, axis=-1)
            cos = np.dot(fa, fb.T) / (l2_a * l2_b.T)
            dis = .5 * np.max(cos) + .5
        elif type == 'l1':
            # l2 distance
            l2_a = np.linalg.norm(fa, axis=-1, keepdims=True)
            l2_b = np.linalg.norm(fb, axis=-1, keepdims=True)
            fa /= l2_a
            fb /= l2_b
            l1_dis = np.square(fa - fb)
            # l1_dis = np.abs(fa - fb)
            l1_dis = np.sum(l1_dis, axis=-1)
            dis = l1_dis # 
        return dis
