import cv2
from videofactory import CVideoFactory
from frameprocess import CProcessEG
from workingconfig import getWorkingConfig


if __name__ == '__main__':
    config = getWorkingConfig()
    # Frame process
    print('initialze process eg!')
    process = CProcessEG()
    process.initialize(config)
    # create and open video device.
    videoFactory = CVideoFactory()
    videoDevice = videoFactory.getVideoGenerator(config['devicename'])
    # open video source
    opened = videoDevice.open(config['videosource'])
    if opened is False:
        print('open video source failed!')
    else:
        frame_size = videoDevice.getSize()
        print('video resolution:{}'.format(frame_size))
        # main loop
        while 1:
            ret, frame = videoDevice.readFrame()
            if ret:
                process.push(frame, frame_size)
                cv2.imshow('show', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        process.uninit()
        videoDevice.close()
        cv2.destroyAllWindows()
        print('Quit!')
