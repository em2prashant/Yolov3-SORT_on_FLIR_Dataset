import json
import numpy as np
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

file_path = 'thermal_annotations.json'

file = json.load(open(file_path))

idd = []
image_names = []

for elements in file['images']:
	image_names.append(elements['file_name'])
	idd.append(elements['id'])

cat_id = {}
bbox = {}

for elements in file['annotations']:
	
	if elements['image_id'] not in cat_id.keys():
		cat_id[elements['image_id']] = []
		cat_id[elements['image_id']].append(elements['category_id'])
	else:
		cat_id[elements['image_id']].append(elements['category_id'])

	if elements['image_id'] not in bbox.keys():
		bbox[elements['image_id']] = []
		bbox[elements['image_id']].append(elements['bbox'])
	else:
		bbox[elements['image_id']].append(elements['bbox'])

'''
with open('gt_det.txt', 'w+') as inp_file:
	for i_d in idd:
		try:
			for i in range(len(cat_id[i_d])):
				print('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,1,-1,-1,-1'%(i_d+1, cat_id[i_d][i], bbox[i_d][i][0], bbox[i_d][i][1], bbox[i_d][i][2], bbox[i_d][i][3]), file = inp_file)
		except:
			pass
'''


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

for i_d, img_name in zip(idd,image_names):
	
	img = plt.imread(img_name)
	
	try:
		boxes = np.array(bbox[i_d])
		cat = cat_id[i_d]
	except:
		pass
	
	ax.imshow(img, cmap = 'gray')
	plt.title(i_d)

	for b,i in zip(boxes,cat):
		ax.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], fill = False, lw = 3))
		if i == 1:
			name = 'Person'
		elif i == 2:
			name = 'Bicycle'
		elif i == 3:
			name = 'Car'
		elif i == 17:
			name = 'Dog'
		plt.text(x = b[0], y = b[1], s = name)
	
	fig.canvas.flush_events()
	plt.draw()
	ax.cla()

