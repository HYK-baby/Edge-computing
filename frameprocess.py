import colorsys
import os
import time
import cv2
import numpy as np
from yolo_lite_run import YOLOv3tflite
from fcos_lite_run import Mydetector
from keyframe import Keyframe


class CDrawEG():
    def __init__(self, class_names):
        self.names = class_names
        self.num = len(class_names)
        # 产生多种颜色
        hsv_tuples = [(x / self.num, 1., 1.) for x in range(self.num)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def drawboexes(self, frame, objects):
        # draw bounding boxes
        for n in range(len(objects)):
            obj = objects[n]
            cid = obj['class']
            lt = (obj['box'][0], obj['box'][1])
            rb = (obj['box'][2], obj['box'][3])
            cv2.rectangle(frame, lt, rb, self.colors[cid])
            # FONT_HERSHEY_PLAIN FONT_HERSHEY_SIMPLEX
            lt = (obj['box'][0], obj['box'][1]+15)
            cv2.putText(frame, self.names[cid], lt, cv2.FONT_HERSHEY_SIMPLEX, .75, self.colors[cid], 1)


        
class CProcessEG():
    def __init__(self):
        self.dispaly = 0
        self.count = 0
        self.objects = []
        self.result_file = None
        self.start_time = 0
        self.keyframe_count = 0        
        self.cur_frame = 0
        self.total_frame = 1000  # every N frames, cal running states
        self.judge_keyframe = True

    def initialize(self, config):
        """
        config: dictionary of params
        """
        if config['judge_keyframe'] == 1:
            self.judge_keyframe = True
        else:
            self.judge_keyframe = False
        
        self.keyframe_path = config['keyframe_path']
        self._prepare_storage(self.keyframe_path)
        # load class names
        with open(config['classnamefile']) as f:
            self.class_names = f.readlines()
        for i in range(len(self.class_names)):
            name = self.class_names[i]
            if name[-1] == '\n':
               name = name[:len(name)-1]
               self.class_names[i] = name
            print(name)
        self.num_classes = len(self.class_names)
        # load model
        score_threshold = config['score_threshold']
        iou_threshold = config['iou_threshold']
        modelname = config['modelname']
        input_shape = config['input_shape']
        print('====load model:{}, inputshape:{}'.format(modelname, input_shape))
        print('====display:{}'.format(config['dispaly']))
        anchor = config['anchor']
        if modelname == 'yolov3':
            self.model = YOLOv3tflite(config['modelpath'], input_shape, self.num_classes, anchor, score_threshold, iou_threshold)
        elif modelname == 'mydetector':
            self.model = Mydetector(config['modelpath'], input_shape, self.num_classes, anchor, score_threshold, iou_threshold)
        # create key frame
        self.keyframe = Keyframe(.2, 'myencode.tflite', input_shape)
        self.keyframe_id = 0

        self.input_shape = input_shape
        self.drawer = CDrawEG(self.class_names)
        self.dispaly = config['dispaly']

        return True
    
    def uninit(self):
        self.result_file.close()
        # self.frame_dis.close()
        
    def _prepare_storage(self, keyframe_path):
        # is storage path exist
        exists = os.path.exists(keyframe_path)
        if exists is not True:
            os.makedirs(keyframe_path)
        run_time = time.localtime(time.time())
        file_name = '/result_%02d_%02d_%02d_%02d_%02d'%(run_time[1],run_time[2],run_time[3],run_time[4],run_time[5])
        self.result_file = open(keyframe_path+file_name+'.txt', 'w')
        # self.frame_dis = open(keyframe_path+file_name+'dis.txt', 'w')

    def saveframe(self, frame, size, objects):
        run_time = time.localtime(time.time())
        file_name = '/img%02d_%02d_%02d_%02d_%02d.jpg'%(run_time[1],run_time[2],run_time[3],run_time[4],run_time[5])
        img_path = self.keyframe_path + file_name
        exists = False
        for i in range(1, 10):
            exists = os.path.exists(img_path)
            if exists:
                file_name = '/img%02d_%02d_%02d_%02d_%02d_%d.jpg'%(run_time[1],run_time[2],run_time[3],run_time[4],run_time[5], i)
                img_path = self.keyframe_path + file_name
            else:
                break
        if exists:
            return
        cv2.imwrite(img_path, frame)
        # save result
        strResult = img_path
        for obj in objects:
            strObj = ' %d,%d,%d,%d,%d'%(obj['class'], obj['box'][0], obj['box'][1], obj['box'][2], obj['box'][3])
            strResult += strObj
            strResult += '\n'
        self.result_file.write(strResult)
        self.result_file.flush()

    def push(self, frame, size):
        self.count += 1
        if self.count > 100000:
            self.count -= 100000
        # if self.count % 2 == 0:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        isKeyframe, objects = self.modelinference(image, size)
        if isKeyframe:
            # save keyframe and result
            # self.saveframe(frame, size, objects)
            self.objects = objects
        # if len(objects) > 0:
        # print(objects)
        if self.dispaly:
            self.drawer.drawboexes(frame, self.objects)
        # save keyframe and result
        if isKeyframe:
            self.saveframe(frame, size, self.objects)
            
    def modelinference(self, image, size):
        """
        model inference. return detection result.
        """
        if self.start_time == 0:
            self.start_time = time.time()
        self.cur_frame += 1
        if self.cur_frame >= self.total_frame:
            self.cur_frame = 0
            end = time.time()
            elaps = end - self.start_time
            time_per_frame = elaps * 1000 / self.total_frame
            fps = 1000.0 / time_per_frame
            print('Running state every {} frames: elaps {:.4}s, {:.4}FPS, keyframe count {}.'.format(self.total_frame, elaps, fps, self.keyframe_count))
            self.start_time = end
            self.keyframe_count = 0

        # 判断是否是关键帧
        isKeyframe = True
        if self.judge_keyframe:
            padded_img, padding_lt = self._resize_image(image, size, (5*32, 6*32))
            isKeyframe, distance = self.keyframe.judge(padded_img)
            # if isKeyframe:
            #     self.keyframe_id += 1
            #     distance_path = './keyframe/distance%d' % self.keyframe_id
            #     np.save(distance_path, distance)
            # print('key frame distance:', distance)
            # self.frame_dis.write('{:.4f} '.format(distance))
        # detection
        if isKeyframe:
            self.keyframe_count += 1
            padded_img, padding_lt = self._resize_image(image, size, self.input_shape)
            padding_lt = padding_lt[::-1]
            # print('padding_lt:', padding_lt)
            start_time = time.time()
            objects = self.model.predicted(padded_img)
            stop_time = time.time()
            
            target_shape = np.array(self.input_shape, dtype=np.float32)
            org_img_shape = np.array(size, dtype=np.float32)
            scale = target_shape / org_img_shape
            scale = scale.min()
            target_shape = target_shape[::-1]
            # if len(objects) > 0:
            #     print('detect {} objects'.format(len(objects)))
            #     print('frame{}, elaps time: {:.3f}ms'.format(self.count, (stop_time - start_time) * 1000))
            for i in range(len(objects)):
                # normalized coordinate to absolute
                boxes_lt = objects[i]['box'][0:2] * target_shape - padding_lt
                boxes_rb = objects[i]['box'][2:4] * target_shape - padding_lt
                boxes_lt /= scale
                boxes_rb /= scale
                box_coord = np.concatenate([boxes_lt, boxes_rb], axis=0)
                # box_coord = box_coord + detal
                box_coord = box_coord.astype(np.int32)
                objects[i]['box'] = box_coord
            return isKeyframe, objects
        else:
            return isKeyframe, []
    
    def _resize_image(self, image, size, inputshape):
        # resize
        org_img_shape = np.array(size, dtype=np.float32)
        input_shape = np.array(inputshape, dtype=np.float32)
        scale = input_shape / org_img_shape
        scale = scale.min()
        aspect_ratio = org_img_shape * scale
        padding_lt = (inputshape - aspect_ratio) / 2
        padding_lt = padding_lt.astype(np.int32)
        # detal = (input_shape - aspect_ratio)/2
        # detal = detal[::-1]  # image row first, detal=[dy,dx] ==> [dx,dy]
        # detal = np.concatenate([detal, detal], axis=0)
        # print('detal:', detal)
        toCoord_shape = org_img_shape[::-1]
        toCoord_shape = np.concatenate([toCoord_shape, toCoord_shape], axis=0)

        # resize image with pad, then predict
        padded_img = np.ones((inputshape[0], inputshape[1], 3), dtype=np.uint8) * 114
        resized_img = cv2.resize(image, (int(aspect_ratio[1]), int(aspect_ratio[0])), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        padded_img[padding_lt[0]:int(aspect_ratio[0]+padding_lt[0]), :int(aspect_ratio[1])] = resized_img
        return padded_img, padding_lt
        
