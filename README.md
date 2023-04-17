# Object Detection for End-Edge-Cloud Video Surveillance


A lightweight object detection model called AT-YOLO is designed to be deployed on Raspberry Pi.
AT-YOLO is trained on the COCO dataset.

Results of COCO2017 test-dev
Model|Input|GFLOPs|Params|AP|AP50|AP75
 --- | --- | ---- | ---- | - | - | --
SSD|300×300|35.2|34.3M|25.1|43.1|25.8
DSSD321|321×321|22.3|-|28.0|45.4|29.3
YOLOv3|416×416|65.9|62.3M|1.0|55.3|32.3
MobileNet-SSDLite|320×320|2.60|5.1M|22.2|-|-
Pelee|304×304|1.21|5.98M|22.4|38.3|22.9
CSPPeleeRef|320×320|3.43|5.67M|23.5|44.6|22.7
Tiny-DSOD|300×300|1.06|1.15M|23.2|40.4|22.8
MoblienetV3+D-head|320×320|2.02|4.77M|25.8|41.6|26.2
AT-YOLO 320|320×320|2.02|3.47M|**26.2**|39.8|26.8
YOLOv3-Tiny|416×416|5.57|8.86M|16.6|-|-
YOLOv4-Tiny|416×416|6.96|6.06M|21.7|-|-
Pelee-PRN|416×416|4.04|3.16M|23.3|45.0|22.0
MoblienetV3+D-head|416×416|3.42|4.77M|29.1|45.9|30.6
AT-YOLO 416|416×416|3.43|3.47M|**29.9**|**46.5**|**32.0**


![The Structure of Smart Video Surveillance](https://github.com/HYK-baby/Edge-computing/blob/main/graph/overview.png)

We designed three modes for comparison, which are with a keyframe algorithm, without a keyframe algorithm, and with object detection on the cloud.
![Bandwidth consumption](https://github.com/HYK-baby/Edge-computing/blob/main/graph/bandwidth.png)
