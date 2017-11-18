import numpy as np
import matplotlib.pyplot as plt
import data_loader as dl
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
from sklearn.preprocessing import label_binarize
import cnn_generator as cg
import os
import helper_functions as hf
from sklearn.metrics import classification_report
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import keras.backend as K
import datetime
import os
import sys
import argparse
import glob

def get_args():
	parser = argparse.ArgumentParser()
	now = datetime.datetime.now()
	date = now.strftime("%Y.%m.%d")
	parser.add_argument("--nb_stage", type=int, default=2)
	parser.add_argument("--experiment", type=str, default='experiment-11.15')
	parser.add_argument("--ku", type=int, default=3)
	parser.add_argument("--group", type=int, default=0)
	parser.add_argument("--img_size", type=int, default=256)
	parser.add_argument("--input_size", type=int, default=100)
	parser.add_argument("--nb_tr_sample", type=int, default=100)
	parser.add_argument("--val_version", type=int, default=16)
	parser.add_argument("--date", type=str, default=date)
	parser.add_argument("--nb_round", type=int, default=4)
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--nb_epoch_per_record", type=int, default=1)
	parser.add_argument("--lr", type=int, default=0.01)
	parser.add_argument("--batch_size", type=int, default=170)
	parser.add_argument("--gpu", type=str, default='0')
	parser.add_argument("--start_stage", type=int, default='0')
	parser.add_argument("--round", type=int, default= 0)	
	args = parser.parse_args()
	return args

args = get_args()
print(args)

gpu = args.gpu
experiment = args.experiment
ku = args.ku
img_size = args.img_size
input_size = args.input_size
nb_tr_sample = args.nb_tr_sample
val_version = args.val_version
nb_epochs = args.nb_epochs
nb_epoch_per_record = args.nb_epoch_per_record
date = args.date
nb_round = args.nb_round
lr=args.lr
batch_size = args.batch_size
val_version =args.val_version
nb_stage = args.nb_stage
round = args.round
start_stage = args.start_stage

import time

## mapping group to GPUs
group = args.group

if group == 0:
	rounds_pool = [0,1]
elif group == 1:
	rounds_pool = [2,3]

root_folder = experiment+'/'
for stage in range(start_stage, nb_stage):
	stage_folder = root_folder + 'stage-'+str(stage)+'/'
	train_index_files = glob.glob(stage_folder+'train*.txt')
	if len(train_index_files)==0:
		if stage >0:
			prev_stage_folder = root_folder + 'stage-'+str(stage-1)+'/'
			delay_time = random.random()*10+5
			while True:
				print('wating for sync files ....')
				sync_files = glob.glob(prev_stage_folder+'sync*')
				if len(sync_files) == nb_round:
					break
				time.sleep(delay_time)
		os.system('python3 suggestion.py --stage '+str(stage)+' --experiment '+experiment+' --gpu '+gpu+' --ku '+str(ku))
# 		os.system('python3 suggestion.py --stage '+str(stage)+' --experiment '+experiment+' --gpu '+gpu+' --ku '+str(ku))
	for round in rounds_pool:
		os.system('python3 activeRes100.py --stage '+str(stage)+' --round '+str(round)+' --experiment '+experiment+' --nb_epochs '+str(nb_epochs)+' --val_version '+str(val_version)+' --gpu '+gpu)
	cur_stage_folder = root_folder + 'stage-'+str(stage)+'/'
	if round == 3:
		while True:
			sync_files = glob.glob(cur_stage_folder+'sync*')
			if len(sync_files)==nb_round:
				print('evaluating the model ....')
				os.system('python3 activeTest100new.py --stage '+str(stage)+' --experiment '+experiment+' --gpu '+gpu+' --val_version '+str(val_version))
				break
			time.sleep(10)

## one button train
# thread_list =[]
# for i in range(nb_round):
# 	thread_list.append(trainThread(experiment, nb_stage, i, str(i), nb_epochs, val_version))
# 
# for i in range(nb_round):
# 	thread_list[i].start()


