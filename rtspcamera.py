import cv2


def get_img_from_camera_net(folder_path):
    print('============Camera==============')
    camera = cv2.VideoCapture("rtsp://192.168.31.152:554/ch01_sub.264")  # 获取网络摄像机
    opened = camera.isOpened()
    if opened is not True:
        print('camera Failed!')
        return 1
    fps = camera.get(cv2.CAP_PROP_FPS)  # 获取帧率
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))  # 一定要转int 否则是浮点数
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('video FPS{}, widht{}, height{}'.format(fps, width, height))
    
    frame_count = 0
    while opened:
        ret, frame = camera.read()
        if ret:
            frame_count = frame_count + 1
            # print(frame_count)
            
            cv2.imshow('show', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
    return 0


# 测试
if __name__ == '__main__':
    folder_path = 'D:\\Anacon'
    get_img_from_camera_net(folder_path)
