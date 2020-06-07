import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_label_text_files', type = str, default = 'labels', help = 'Enter the path to the labels folder')
args = parser.parse_args()

names = sorted(os.listdir(args.path_to_label_text_files))

# Out of 8862 images in FLIR 'train' folder, only 7860 images have labels
# Out of 1366 images in FLIR 'val' folder, only 1354 images have labels

train_names = names[0:7860]
valid_names = names[7860:]

with open('train.txt', 'w+') as train_file:
	for i in range(len(train_names)):
		print(train_names[i][:-4] + '.jpeg', file = train_file)

with open('valid.txt', 'w+') as val_file:
	for i in range(len(valid_names)):
		print(valid_names[i][:-4] + '.jpeg', file = val_file)