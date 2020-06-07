import os
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_train_annotations', type = str, default = 'thermal_annotations.json', help = 'Enter the path to the training thermal annotations file') 
parser.add_argument('--path_to_val_annotations', type = str, default = 'thermal_annotations.json', help = 'Enter the path to the validation thermal annotations file')
args = parser.parse_args()

train_ann_path = args.path_to_train_annotations
val_ann_path = args.path_to_val_annotations

file_train = json.load(open(train_ann_path))
file_val = json.load(open(val_ann_path))

def write_ann_to_text_files(file, offset):
	
	cat_id = {}
	bbox = {}
	image_names = []
	cat_names = []
	cat_idd = []

	for element in file['categories']:
		cat_names.append(element['name'])
		cat_idd.append(element['id'])

	for elements in file['images']:
		image_names.append(elements['file_name'][-15:24])

	for elements in file['annotations']:
		if int(elements['category_id']) == 1: # For Person
			cat = 0
		elif int(elements['category_id']) == 2: # For Bicycle
			cat = 1
		elif int(elements['category_id']) == 3: # For Cars
			cat = 2
		elif int(elements['category_id']) == 17: # For Dogs
			cat = 3

		if elements['image_id'] not in cat_id.keys():
			cat_id[elements['image_id']] = []
			cat_id[elements['image_id']].append(cat)
		else:
			cat_id[elements['image_id']].append(cat)

		if elements['image_id'] not in bbox.keys():
			bbox[elements['image_id']] = []
			bbox[elements['image_id']].append(elements['bbox'])
		else:
			bbox[elements['image_id']].append(elements['bbox'])

	im_id = []
	for elements in file['annotations']:
		if elements['image_id'] in im_id:
			pass
		else:
			im_id.append(elements['image_id'])

	os.makedirs('labels/', exist_ok = True)

	for filename in image_names:
		image_id = int(filename[5:10]) - 1 - offset
		if image_id in im_id:
			with open('labels/' + filename + '.txt', 'w+') as ann_file:
				box = np.array(bbox[image_id])
				cat = np.array(cat_id[image_id]).reshape(len(cat_id[image_id]),1)
				towrite = np.concatenate((cat, box), axis = 1)
				for i in range(towrite.shape[0]):
					x_centre = (towrite[i,1] + towrite[i,3]/2)/640
					y_centre = (towrite[i,2] + towrite[i,4]/2)/512
					width = towrite[i,3]/640
					height = towrite[i,4]/512
					print('%f %f %f %f %f'%(towrite[i,0],x_centre,y_centre,width,height), file = ann_file)
		else:
			pass

write_ann_to_text_files(file_train, offset = 0)
write_ann_to_text_files(file_val, offset = 8862)