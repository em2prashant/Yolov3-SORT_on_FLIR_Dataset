# Yolov3+SORT_on_FLIR_Dataset

## Authors:
<a href = "https://github.com/em2prashant">Prashant Agrawal</a><br>
<a href = "https://github.com/jainpulkit54">Pulkit Jain</a>

## Description:
In this repository, a <a href = "https://pjreddie.com/media/files/papers/YOLOv3.pdf">"YOLOv3: An Incremental Improvement"</a> detector has been trained on the open sourced <a href = "https://www.flir.in/oem/adas/adas-dataset-form/">FLIR Dataset</a>. Apart from training the detector, the task was also to track the detected objects which was carried out using the <a href = "https://arxiv.org/pdf/1602.00763.pdf">"Simple Online and Realtime Tracking"</a> algorithm.
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

## Output Video:
The tracking output video can be found at the following link:
Link: https://drive.google.com/file/d/1ICfcUHrX_Cd3yBGXYq2EE8HUF4-W94Rp/view?usp=sharing

## Yolov3 Detector weights:
For Yolov3 @ 608:
Link: https://drive.google.com/file/d/19x6B6oYB8Y4o3Sqp5YXAPBEI38yP2Uq9/view?usp=sharing

For Yolov3 @ 416:
Link: https://drive.google.com/file/d/1M2xr2aW0Zj1ghM91FF4EoBgwCQpRNz2p/view?usp=sharing

## How to run the code:

In order to run the code:

1) The detections need to be saved to a text file which can be done by running the "YOLOv3 detector".<br>
<code>python detect.py --image_folder data/flir/ --images_path data/flir/images/ --model_def config/yolov3-custom_flir.cfg --weights_path checkpoints/model_param_flir_608.pth --class_path data/flir/classes.names --text_file_path data/flir/det_yolov3_608/det_yolov3_608.txt --conf_thresh 0.8 --nms_thresh 0.5</code> <br>
2) Once the detections have been stored in a text file, the next step is to run the "SORT tracker" code with the stored detections and save those images in a folder.<br>
<code>python tracker.py --display True --path_to_detections data/flir/det_yolov3_608/det_yolov3_608.txt --path_to_images data/flir/images/ --path_to_store_trackers output/trackers/yolov3_608.txt --path_to_store_images_with_detections output/detections/</code>
3) After the images with trackers are stored in a folder, the last step will be to make a video out of it. The code provided here makes the video at the desired FPS (10 FPS recommended).<br>
<code>python make_video.py --output_image_path output/detections/ --output_video_name flir_output_video.avi --fps 10</code>

