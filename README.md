# Yolov3+SORT_on_FLIR_Dataset

## Description:
In this repository, a <a href = "https://pjreddie.com/media/files/papers/YOLOv3.pdf">"YOLOv3: An Incremental Improvement"</a> detector has been trained on the open sourced <a href = "https://www.flir.in/oem/adas/adas-dataset-form/">FLIR Dataset</a>. Apart from training a detector, the task was also to track the detected objects which was carried out using the <a href = "https://arxiv.org/pdf/1602.00763.pdf">"Simple Online and Realtime Tracking"</a> algorithm.
The mAP obtained for the "YOLOv3" detector trained on the "FLIR Dataset" is tabulated as follows:
<table border = "1">
  <tr>
    <th> Class Number </th>
    <th> Class Name </th>
    <th> Average Precision </th>
  </tr>
  <tr>
    <td>0</td>
    <td>Person</td>
    <td>0.60665</td>
  </tr>
  <tr>
    <td>1</td>
    <td>bicycle</td>
    <td>0.38672</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Car</td>
    <td>0.76392</td>
  </tr>
  <tr>
    <td>3</td>
    <td>Dog</td>
    <td>0.00420</td>
  </tr>
</table>

The overall mAP obtained for 4 classes is: 0.44037 <br>
The overall mAP obtained for the 3 dominant classes is: 0.58576 <br>
The reason for low AP for the "Dog" class is the low presence of this class in the dataset as it is evident from the desciption of the FLIR Dataset.

## Dataset:
Link: https://www.flir.in/oem/adas/adas-dataset-form/ <br>
This dataset basically provides three types of thermal images: <br>
1) train: Contains 8862 thermal images <br>
2) val: Contains 1366 thermal images <br> 
3) video: Contains 4224 thermal images <br>

The images present in the "train" and "val" folder were used as the "train" and "validation" sets respectively for training the "YOLOv3" detector.
