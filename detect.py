import os
import time
import torch
import argparse
import numpy as np

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models import *
from utils.utils import *
from skimage import io
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as patches
matplotlib.use('TkAgg')

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type = str, default = "data/flir/", help = "path to dataset")
parser.add_argument("--images_path", type = str, default = "data/flir/images/", help = "Path to the images")
parser.add_argument("--text_file_path", type = str, default = "data/flir/det_yolov3_608/det_yolov3_608.txt", help = "This is the text file where the detections will be stored in MOT format")
parser.add_argument("--model_def", type = str, default = "config/yolov3-custom_flir.cfg", help = "path to model definition file")
parser.add_argument("--weights_path", type = str, default = "checkpoints/model_param_flir_608.pth", help = "path to weights file")
parser.add_argument("--class_path", type = str, default = "data/flir/classes.names", help = "path to class label file")
parser.add_argument("--conf_thres", type = float, default = 0.8, help = "object confidence threshold")
parser.add_argument("--nms_thres", type = float, default = 0.5, help = "iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type = int, default = 1, help = "size of the batches")
parser.add_argument("--n_cpu", type = int, default = 0, help = "number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type = int, default = 416, help = "size of each image dimension")
parser.add_argument("--display", type = str, default = 'False', help = "Set this to True if want to visualize detections on image")
opt = parser.parse_args()

if opt.display == 'False':
	opt.display = False
else:
	opt.display = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('output', exist_ok = True)

model = Darknet(opt.model_def, img_size = opt.img_size)
model.to(device)

# Loading the checkpoint weights
model.load_state_dict(torch.load(opt.weights_path))
model.eval()

images = ImageFolder(
	opt.image_folder,
	transform = transforms.Compose([transforms.Resize(size = (opt.img_size, opt.img_size)), transforms.ToTensor()])
	)

all_images_names = sorted(os.listdir(opt.images_path))

'''
nums = []
for img_name in all_images_names:
	num = int(img_name[:-5])
	nums.append(num)

nums = sorted(nums)

all_images_names = []
for num in nums:
	name = str(num) + '.png'
	all_images_names.append(name)

#print(all_images_names)
'''

dataloader = DataLoader(images, batch_size = 1, shuffle = False, num_workers = 0)
classes = load_classes(opt.class_path)

# This is the text file to store the detections in MOT format to be used by the SORT tracker
text_file_path = opt.text_file_path
file_det = open(text_file_path, 'w+')

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
#Tensor = torch.FloatTensor
print('Performing Object Detection\n')

if opt.display:
	plt.ion() # Interative Mode On
	fig = plt.figure()
	ax = fig.add_subplot(111)

for index, (image, _ ) in enumerate(dataloader):
	
	img = image.type(Tensor)
	orig_image = Image.open(opt.images_path + all_images_names[index])
	orig_image_size = orig_image.size
	
	if opt.display:
		ax.imshow(orig_image, cmap = 'gray')
		plt.title('Detected Objects in Frame ' + str(index+1))
	
	tic = time.time()
	
	with torch.no_grad():
		detections = model(img)
		detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

	toc = time.time()
	#print('Processing Time',toc - tic)

	try:
		detections = detections[0].numpy()
		# The detections contain 7 columns which are [x_min, y_min, x_max, y_max, obj_conf, cls_conf, cls_num]
		# and the bounding box coordinates are in 'opt.img_size' scale 
		unique_classes = np.unique(detections[:,-1])
		n_detections = detections.shape[0]
		colors = np.random.rand(n_detections,3)

		for d_num, (x_min, y_min, x_max, y_max, obj_conf, cls_conf, cls_num) in enumerate(detections):
			
			x_min, y_min, x_max, y_max = x_min/opt.img_size, y_min/opt.img_size, x_max/opt.img_size, y_max/opt.img_size
			x_min = x_min * orig_image_size[0]
			y_min = y_min * orig_image_size[1]
			x_max = x_max * orig_image_size[0]
			y_max = y_max * orig_image_size[1]
			width = x_max - x_min
			height = y_max - y_min
			cls_num = int(cls_num)

			if cls_num == 0:
				name = classes[0]
			elif cls_num == 1:
				name = classes[1]
			elif cls_num == 2:
				name = classes[2]
			elif cls_num == 3:
				name = classes[3]

			if opt.display:
				ax.add_patch(patches.Rectangle((x_min, y_min), width, height, fill = False, lw = 1, ec = colors[d_num, :]))
				plt.text(x = x_min, y = y_min, s = name + str('%0.2f'%cls_conf))
			
			file_det.write(str(index+1)+','+str(cls_num)+','+str('%0.2f'%x_min)+','+str('%0.2f'%y_min)+','+str('%0.2f'%width)+','+str('%0.2f'%height)+','+str('%0.2f'%cls_conf)+','+'-1'+','+'-1'+','+'-1')
			file_det.write('\n')

	except:
		pass

	if opt.display:
		fig.canvas.flush_events()	
		plt.draw()
		ax.cla()