# Yolov3+SORT_on_FLIR_Dataset

## Description:
In this repository, a <a href = "https://pjreddie.com/media/files/papers/YOLOv3.pdf">"YOLOv3: An Incremental Improvement"</a> detector has been trained on the open sourced <a href = "https://www.flir.in/oem/adas/adas-dataset-form/">FLIR Dataset</a>. Apart from training a detector, the task was also to track the detected objects which was carried out using the <a href = "https://arxiv.org/pdf/1602.00763.pdf">"Simple Online and Realtime Tracking"</a> algorithm.
The mAP obtained for the "YOLOv3" detector trained on the "FLIR Dataset" is tabulated as follows:

|Model | Class Number | Class Name | Average Precision|
|------|--------------|------------|------------------|
||0|Person|0.49848|
|Yolov3 @ 416|1|Bicycle|0.31797|
||2|Car|0.70044|
||3|Dog|0.00446|

The overall mAP obtained for 4 classes is: 0.38033 <br>
The overall mAP obtained for the 3 dominant classes is: 0.50563 <br>

|Model | Class Number | Class Name | Average Precision|
|------|--------------|------------|------------------|
||0|Person|0.60665|
|Yolov3 @ 608|1|Bicycle|0.38672|
||2|Car|0.76392|
||3|Dog|0.00420|

The overall mAP obtained for 4 classes is: 0.44037 <br>
The overall mAP obtained for the 3 dominant classes is: 0.58576 <br>

The reason for low AP for the "Dog" class is the low presence of this class in the dataset as it is evident from the desciption of the FLIR Dataset. Also, this repository uses "YOLOv3 @ 608" for detecting objects as well as tracking.

## Dataset:
Link: https://www.flir.in/oem/adas/adas-dataset-form/ <br>
This dataset basically provides three types of thermal images: <br>
1) train: Contains 8862 thermal images <br>
2) val: Contains 1366 thermal images <br> 
3) video: Contains 4224 thermal images <br>

The images present in the "train" and "val" folder were used as the "train" and "validation" sets respectively for training the "YOLOv3" detector and the mAP reported above is on the "validation" set of the FLIR Dataset. For the purpose of tracking the detected objects, the "video" folder of the FLIR Dataset is used.

## How to run the code:


