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
from keras.models import load_model
import scipy.misc as misc
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import keras.backend as K
import sys
import argparse


# def get_session(gpu_fraction=0.9):
# #     '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
#     num_threads = os.environ.get('OMP_NUM_THREADS')
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#     if num_threads:
#         return tf.Session(config=tf.ConfigProto(
#             gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
#     else:
#         return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# 
# KTF.set_session(get_session())

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--stage", type=int, default=1)
	parser.add_argument("--experiment", type=str, default='experiment-11.14')
	parser.add_argument("--img_size", type=int, default=256)
	parser.add_argument("--input_size", type=int, default=100)
	parser.add_argument("--nb_tr_sample", type=int, default=100)
	parser.add_argument("--val_version", type=int, default=16)
	parser.add_argument("--gpu", type=str, default='0')
	parser.add_argument("--train_include", type=int, default=0)
	args = parser.parse_args()
	return args

args = get_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# stage = 1
# experiment = 'experiment-11.14'
# ku = 3
# img_size = 256
# input_size = 100
# nb_tr_sample =100
# val_version = 15

stage = args.stage
experiment = args.experiment
input_size = args.input_size
nb_tr_sample = args.nb_tr_sample
val_version = args.val_version
train_include =args.train_include
image_size = args.img_size
model_root = args.experiment+'/'   ## the model_root folder is the experiment name


## data load and preprocess	
X_train,X_val,Y_train,Y_val=hf.load_random_data(val_version = val_version)
if train_include == 1:
	X_val = np.concatenate([X_train, X_val], axis =0)
	Y_val = np.concatenate([Y_train, Y_val], axis =0)

image_patches = hf.image_depatch(X_val, 100)
density_patches = hf.image_depatch(Y_val, 100)
shp = image_patches.shape
image_patch_arr = image_patches.reshape(shp[0],shp[1],shp[2],1)
density_patch_arr = density_patches.reshape(shp[0],shp[1],shp[2],1)

import glob
## metrics
mean_abs_error1 = 0
std_abs_error1 = 0
ave_estimate1 = 0
ave_real1 = 0
ave_acc1 = 0

## uncertain selection strategy
model_folders = glob.glob(model_root+'stage-'+str(stage)+'/'+'round*')
if len(model_folders)>0:
	all_map_ls = []
	# predictions
	for folder in model_folders:
		K.clear_session()
		model_folder = folder+'/model.h5'
	# 	print(model_folder)
		model = load_model(model_folder)
		preds = model.predict(image_patch_arr).reshape(shp[0],shp[1],shp[2])
		preds = preds/100
		estimated_maps = hf.image_merge(preds,image_size)
		all_map_ls.append(estimated_maps)

	all_map_arr = np.array(all_map_ls)
	rot_all_map = np.transpose(all_map_arr, (1,2,3,0))
	ave_estimations = np.mean(rot_all_map, axis = 3)
	estimated_counts = np.apply_over_axes(np.sum,ave_estimations,[1,2]).reshape(estimated_maps.shape[0])
	real_counts = np.apply_over_axes(np.sum,Y_val,[1,2]).reshape(estimated_maps.shape[0])
	mean_abs_error1 = np.mean(np.abs(estimated_counts-real_counts))
	std_abs_error1 = np.std(np.abs(estimated_counts-real_counts))
	ave_estimate1 = np.mean(estimated_counts)
	ave_real1 = np.mean(real_counts)
	ave_acc1 = (ave_real1-mean_abs_error1)/ave_real1
	# print(np.mean(estimated_counts),np.mean(real_counts))
	print(mean_abs_error1,std_abs_error1)
	print('image prediction accuracy-->'+str(ave_acc1))

## random selection strategy
mean_abs_error2 = 0
std_abs_error2 = 0
ave_estimate2 = 0
ave_real2 = 0
ave_acc2 = 0
model_folders = glob.glob(model_root+'stage-'+str(stage)+'/'+'random_round*')
if len(model_folders)>0:
	all_map_ls = []
	# predictions
	for folder in model_folders:
		K.clear_session()
		model_folder = folder+'/model.h5'
	# 	print(model_folder)
		model = load_model(model_folder)
		preds = model.predict(image_patch_arr).reshape(shp[0],shp[1],shp[2])
		preds = preds/100
		estimated_maps = hf.image_merge(preds,image_size)
		all_map_ls.append(estimated_maps)

	all_map_arr = np.array(all_map_ls)
	rot_all_map = np.transpose(all_map_arr, (1,2,3,0))
	ave_estimations = np.mean(rot_all_map, axis = 3)
	estimated_counts = np.apply_over_axes(np.sum,ave_estimations,[1,2]).reshape(estimated_maps.shape[0])
	real_counts = np.apply_over_axes(np.sum,Y_val,[1,2]).reshape(estimated_maps.shape[0])
	mean_abs_error2 = np.mean(np.abs(estimated_counts-real_counts))
	std_abs_error2 = np.std(np.abs(estimated_counts-real_counts))
	ave_estimate2 = np.mean(estimated_counts)
	ave_real2 = np.mean(real_counts)
	ave_acc2 = (ave_real2-mean_abs_error2)/ave_real2
	# print(np.mean(estimated_counts),np.mean(real_counts))
	print(mean_abs_error2,std_abs_error2)
	print('image prediction accuracy-->'+str(ave_acc2))

## save the results in file
metric_file = model_root + 'result.csv'
import csv
if stage ==0:
	with open(metric_file, 'w') as csvfile:
		fieldnames = ['stage','average_count', 'average_ground', 'error_mean', 'error_std', 'accuracy']
		writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
		writer.writeheader()
		writer.writerow({'stage':stage, 'average_count':ave_estimate1, 'average_ground': ave_real1, 'error_mean': mean_abs_error1, 'error_std': std_abs_error1, 'accuracy':ave_acc1})
		writer.writerow({'stage':stage, 'average_count':ave_estimate2, 'average_ground': ave_real2, 'error_mean': mean_abs_error2, 'error_std': std_abs_error2, 'accuracy':ave_acc2})
else:
	with open(metric_file, 'a') as csvfile:
		fieldnames = ['stage','average_count', 'average_ground', 'error_mean', 'error_std', 'accuracy']
		writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
		# 		writer.writeheader()
		writer.writerow({'stage':stage, 'average_count':ave_estimate1, 'average_ground': ave_real1, 'error_mean': mean_abs_error1, 'error_std': std_abs_error1, 'accuracy':ave_acc1})
		writer.writerow({'stage':stage, 'average_count':ave_estimate2, 'average_ground': ave_real2, 'error_mean': mean_abs_error2, 'error_std': std_abs_error2, 'accuracy':ave_acc2})
		

