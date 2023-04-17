

detectionConfig = {
    'devicename': 'rtspCamera',
    # 'videosource': 'rtsp://192.168.31.152:554/ch01_sub.264',  # rtsp://192.168.31.152:554/ch01_sub.264 C:/Users/youlongbu621/Pictures/视频项目/IMG_2938.MP4
    'videosource': 'IMG_3815.MOV',
    'dispaly': 1,
    # model config
    # 'classnamefile': 'voc_classes.txt',
    'classnamefile': 'coco_classes.txt',
    'input_shape': [256, 320],
    # 'modelname': 'yolov3',
    'modelname': 'mydetector',
    # 'modelpath': 'mydensenet_det2.tflite',
    # 'modelpath': 'mydetector.tflite',
    'modelpath': 'mydensenet_det-panet-f.tflite',
    'score_threshold': .4,
    'iou_threshold': .4,
    'anchor': [[[81, 82], [135, 169], [344, 319]], [[10, 14], [23, 27], [37, 58]]],
    # network config
    'destip': '192.168.31.1',
    'resultport': 5555,
    'videoport': 5554,
    # key frame config
    'judge_keyframe': 1,
    'keyframe_path': './keyframe'
}


def getWorkingConfig():
    return detectionConfig
