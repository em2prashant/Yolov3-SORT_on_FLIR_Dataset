import os
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import matplotlib
matplotlib.use('TkAgg')

def hungarian_algorithm(cost_matrix):

	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	total_cost = cost_matrix[row_ind, col_ind].sum()

	return np.array(list(zip(row_ind, col_ind)))

def iou_individual(ground_truth, prediction):

	# This function calculates the intersection over union between two boxes and returns the same

	x1 = max(ground_truth[0], prediction[0])
	y1 = max(ground_truth[1], prediction[1])
	x2 = min(ground_truth[2], prediction[2])
	y2 = min(ground_truth[3], prediction[3])

	intersection_area = max(0,(x2 - x1)) * max(0,(y2 - y1))
	area_gt = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])
	area_pred = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
	union_area = area_gt + area_pred - intersection_area + 1e-16
	iou = intersection_area/union_area

	return iou

def xyxy2xywh(bbox):

	# This function takes bounding box in the form of [x_min, y_min, x_max, y_max] and
	# returns it in the form of [x_centre, y_centre, width, height]

	x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
	x_centre = (x_min + x_max)/2
	y_centre = (y_min + y_max)/2
	width = x_max - x_min
	height = y_max - y_min

	return np.array([x_centre, y_centre, width, height])

def xyxy2xysr(bbox):

	# This function takes bounding box in the form of [x_min, y_min, x_max, y_max]
	# and then converts it into form of [x_centre, y_centre, scale/area, aspect_ratio]
	
	x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]

	width = x_max - x_min
	height = y_max - y_min
	area = width * height
	aspect_ratio = width/height
	x_centre = x_min + width/2
	y_centre = y_min + height/2

	return np.array([x_centre, y_centre, area, aspect_ratio]).reshape(4,1)

def xysr2xyxy(bbox):

	# This function takes bounding box in the form of [x_centre, y_centre, scale/area, aspect_ratio]
	# and converts it to the form of [x_min, y_min, x_max, y_max]

	x_centre, y_centre, area, aspect_ratio = bbox[0,0], bbox[1,0], bbox[2,0], bbox[3,0]
	height = np.sqrt(area/aspect_ratio)
	width = height * aspect_ratio

	x_min = x_centre - width/2
	x_max = x_centre + width/2
	y_min = y_centre - height/2
	y_max = y_centre + height/2

	return np.array([x_min, y_min, x_max, y_max])

class KalmanFilterTracker(object):

	count = 0
	# The function below initializes the Kalman Filter tracker using the initial bounding box
	# Using a constant velocity model

	def __init__(self, bbox):
		# dim_x --> Represents the dimension of the State Vector
		# dim_z --> Represents the dimension of the Measurement Vector

		# x is the Filter State Estimate and will be of size [dim_x, 1] and default value is np.zeros([dim_x, 1])
		# F matrix is the State Transition Matrix and will be of size [dim_x, dim_x]
		# B matrix is the Control Transition Matrix and will be of size [dim_x, dim_u] and default is 0
		# H matrix is the Measurement Function and will be of size [dim_z, dim_x]
		# R matrix denotes the Measurement Noise/Uncertainity will be of size [dim_z, dim_z] and default is np.eye(dim_z)
		# P matrix denotes the Process Covariance Matrix will be of size [dim_x, dim_x] and default is np.eye(dim_x)
		# Q matrix denotes the Process Noise/Uncertainity Covariance Matrix and will be of size [dim_x, dim_x] and default is np.eye(dim_x)

		self.kf = KalmanFilter(dim_x = 7, dim_z = 4)
		self.kf.F = np.array([
			[1,0,0,0,1,0,0],
			[0,1,0,0,0,1,0],
			[0,0,1,0,0,0,1],
			[0,0,0,1,0,0,0],
			[0,0,0,0,1,0,0],
			[0,0,0,0,0,1,0],
			[0,0,0,0,0,0,1]
			])
		self.kf.H = np.array([
			[1,0,0,0,0,0,0],
			[0,1,0,0,0,0,0],
			[0,0,1,0,0,0,0],
			[0,0,0,1,0,0,0]
			])

		self.kf.R[2:, 2:] *= 10
		self.kf.P[4:, 4:] *= 1000 # Giving high uncertainties to the unobservable initial velocities
		self.kf.P *= 10
		self.kf.Q[-1,-1] *= 0.01 # This modifies the last element in the 2D array
		self.kf.Q[4:,4:] *= 0.01

		self.kf.x[:4] = xyxy2xysr(bbox) # Initialized the Filter State with the bounding box coordinates
		self.time_since_update = 0
		self.id = KalmanFilterTracker.count
		KalmanFilterTracker.count += 1
		self.history = []
		self.hits = 0
		self.hit_streak = 0
		self.age = 0

	def update(self, bbox):

		# Update the Filter State vector with the observed bounding box coordinates
		self.time_since_update = 0
		self.history = []
		self.hits += 1
		self.hit_streak += 1
		self.kf.update(xyxy2xysr(bbox))

	def predict(self):

		# This function advances the State Estimate Vector and returns the predicted bounding box estimate
		if ((self.kf.x[2] + self.kf.x[6]) <= 0):
			self.kf.x[6] *= 0.0
		self.kf.predict()
		self.age += 1
		if self.time_since_update > 0:
			self.hit_streak = 0
		self.time_since_update += 1
		self.history.append(xysr2xyxy(self.kf.x))
		return self.history[-1]

	def get_state(self):

		# This function just returns the current bounding box estimate
		return xysr2xyxy(self.kf.x)

def associate_detections_to_targets(detections, trackers, iou_thresh = 0.3):

	# This function assigns the detections to the tracked objects and both are defined in terms of bounding boxes
	# This function returns 3 lists namely:
	# matched_detections, unmatched_detections and unmatched_targets

	no_of_detections = len(detections)
	no_of_trackers = len(trackers)

	if len(trackers) == 0:
		return np.empty([0,2], dtype = np.int32), np.arange(no_of_detections), np.empty([0,5], dtype = np.int32)

	iou_matrix = np.zeros([no_of_detections, no_of_trackers], dtype = np.float32)

	for index_det, detection in enumerate(detections):
		for index_track, tracker in enumerate(trackers):
			iou_matrix[index_det, index_track] = iou_individual(detection, tracker)

	if min(iou_matrix.shape) > 0:
		a = (iou_matrix > iou_thresh).astype(np.int32)
		if a.sum(axis = 1).max() == 1 and a.sum(axis = 0).max() == 1:
			matched_indices = np.stack(np.where(a), axis = 1)
		else:
			matched_indices = hungarian_algorithm(-1*iou_matrix)
	else:
		matched_indices = np.empty([0,2], dtype = np.int32)

	unmatched_detections = []
	for index_det, detections in enumerate(detections):
		if index_det not in matched_indices[:,0]:
			unmatched_detections.append(index_det)

	unmatched_trackers = []
	for index_track, trackers in enumerate(trackers):
		if index_track not in matched_indices[:,1]:
			unmatched_trackers.append(index_track)

	# Filtering out the matches having low IOU compared to the iou_thresh

	matches = []
	for m in matched_indices:
		if (iou_matrix[m[0], m[1]] < iou_thresh):
			unmatched_detections.append(m[0])
			unmatched_trackers.append(m[1])
		else:
			matches.append(m.reshape(1,2))

	if len(matches) == 0:
		matches = np.empty([0,2], dtype = np.int32)
	else:
		matches = np.concatenate(matches, axis = 0) # This command basically flattens the 2D array into 1D array

	return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class SORT(object):

	def __init__(self, max_age = 1, min_hits = 3):

		# Setting the key parameters for SORT algorithm to work
		self.max_age = 1
		self.min_hits = min_hits
		self.trackers = []
		self.frame_count = 0

	def update(self, dets = np.empty([0,5])):

		# dets = a numpy array of detections in the format [[x_min, y_min, x_max, y_max, score], [x_min, y_min, x_max, y_max, score], ......]
		# It is required that this method be called once for each frame even with empty detections i.e., use np.empty([0,5]) for frames without any detections
		# It returns a similar array where the last column is the object ID.

		# NOTE: The number of objects may differ from the number of detections provided
		self.frame_count = self.frame_count + 1
		# Getting the predicted locations from the existing trackers
		tracks = np.zeros([len(self.trackers), 5])
		to_delete = []
		ret = []
		for index_track, track in enumerate(tracks):
			pos = self.trackers[index_track].predict()
			track[:] = [pos[0], pos[1], pos[2], pos[3], 0]
			if np.any(np.isnan(pos)):
				to_delete.append(index_track)
		tracks = np.ma.compress_rows(np.ma.masked_invalid(tracks))
		for track in reversed(to_delete):
			self.trackers.pop(track)

		matched, unmatched_dets, unmatched_trks = associate_detections_to_targets(dets, tracks)

		# Updating matched trackers with the assigned detections
		for m in matched:
			self.trackers[m[1]].update(dets[m[0],:])

		# Create and initialize new trackers for unmatched detections
		for i in unmatched_dets:
			track = KalmanFilterTracker(dets[i, :])
			self.trackers.append(track)
		i = len(self.trackers)
		for track in reversed(self.trackers):
			d = track.get_state()

			#if (track.time_since_update < 1) and (track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
			ret.append(np.concatenate((d, [track.id + 1])).reshape(1, -1))

			i -= 1
			# Removing the dead tracklet
			if (track.time_since_update > self.max_age):
				self.trackers.pop(i)

		if len(ret) > 0:
			return np.concatenate(ret)
		else:
			return np.empty([0,5])

def parse_args():

	parser = argparse.ArgumentParser(description = 'SORT Demo')
	parser.add_argument('--display', type = str, default = 'False', help = 'Set this to true if want to visualize trackers on videos')
	parser.add_argument('--path_to_detections', type = str, default = 'data/flir/det_yolov3_608/det_yolov3_608.txt', help = 'Specify the path where detections are stored')
	parser.add_argument('--path_to_images', type = str, default = 'data/flir/images/', help = 'Specify the path where the images are stored')
	parser.add_argument('--path_to_store_trackers', type = str, default = 'output/trackers/yolov3_608.txt', help = 'Specify the path to text file where trackers output will be stored')
	parser.add_argument('--path_to_store_images_with_detections', type = str, default = 'output/detections/', help = 'Specify the path to store the images with the detections')
	args = parser.parse_args()
	return args

if __name__ == '__main__':

	args = parse_args()
	
	if args.display == 'False':
		args.display = False
	else:
		args.display = True

	display = args.display
	dir_path = args.path
	total_frames = 0
	total_time = 0.0
	colors = np.random.rand(256,3) # used only for displaying the bounding box trackers

	if display:
		plt.ion()
		fig = plt.figure()
		ax = fig.add_subplot(111)

	det_path = args.path_to_detections
	mot_tracker = SORT() # Creating instance of SORT tracker
	data = np.loadtxt(det_path, dtype = np.float32, delimiter = ',') # Loading all detections of a particular dataset

	with open(args.path_to_store_trackers, 'w+') as file_out:
		print('Currently Processing ' + dataset_name)
		for frame in range(int(data[:,0].max())):
			frame = frame + 1 # Since the frame begin at 1
			dets = data[data[:,0] == frame, 2:7]
			dets[:,2:4] = dets[:,2:4] + dets[:,0:2] # Converting [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max] 
			total_frames = total_frames + 1

			if display:

				image_path = args.path_to_images + '/FLIR_video_%05d.jpeg'%frame
				im = io.imread(image_path)
				ax.imshow(im, cmap = 'gray')
				plt.title('Tracked Targets in ' + dataset_name + ' in frame ' + str(frame))

			tic = time.time()
			trackers = mot_tracker.update(dets)
			toc = time.time()
			total_time = total_time + (toc - tic)

			for d in trackers:
				print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file = file_out)
				if display:
					d = d.astype(np.int32)
					ax.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill = False, lw = 2, ec = colors[d[4]%256, :]))
					plt.text(x = d[0], y = d[1], s = d[4])

			if display:
				plt.savefig(args.path_to_store_images_with_detections + '/FLIR_output_video_%05d.png'%frame)
				fig.canvas.flush_events()
				plt.draw()
				ax.cla()

	print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))