import cv2


class CVideoFactory():
    """
    视频流工厂
    """
    def getVideoGenerator(self, name):
        if name == 'rtspCamera':
            print('device name:', name)
            return CrtspCamera()
        elif name == 'localfile':
            return 0


class CrtspCamera():
    """
    使用OpenCV 通过RTSP协议，读取视频流.
    """
    def __init__(self):
        self.isopen = False
        self.width = self.height = 0, 0

    def getSize(self):
        return (self.height, self.width)

    def open(self, url):
        if self.isopen:
            return self.isopen
        cap = cv2.VideoCapture(url)  # 获取网络摄像机
        if cap.isOpened():
            self.camera = cap
            self.isopen = True
            self.fps = self.camera.get(cv2.CAP_PROP_FPS)  # 获取帧率
            self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))  # 一定要转int 否则是浮点数
            self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return self.isopen

    def close(self):
        if self.isopen:
            self.camera.release()
            self.isopen = False

    def readFrame(self):
        if self.isopen:
            ret, frame = self.camera.read()
            return ret, frame
        else:
            return False, None
