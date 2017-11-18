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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_session(gpu_fraction=0.9):
#     '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())


val_version = 15
train_include =0
image_size = 256
# model_path = '10.10-lr-0.005-scaled-batch-100-v12-FCN1/'+ 'model.h5'
# val_version = 12
	
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
model_root = 'experiment-11.13/'
stage =2

## uncertain selection strategy
model_folders = glob.glob(model_root+'stage'+str(stage)+'/'+'round*')
all_map_ls = []
# predictions
for folder in model_folders:
	K.clear_session()
	model_folder = folder+'/model.h5'
	print(model_folder)
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
mean_abs_error = np.mean(np.abs(estimated_counts-real_counts))
std_abs_error = np.std(np.abs(estimated_counts-real_counts))
print(np.mean(estimated_counts),np.mean(real_counts))
print(mean_abs_error,std_abs_error)
print('image prediction accuracy-->'+str((np.mean(real_counts)-mean_abs_error)/np.mean(real_counts)))

## random selection strategy
model_folders = glob.glob(model_root+'stage'+str(stage)+'/'+'random_round*')
all_map_ls = []
# predictions
for folder in model_folders:
	K.clear_session()
	model_folder = folder+'/model.h5'
	print(model_folder)
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
mean_abs_error = np.mean(np.abs(estimated_counts-real_counts))
std_abs_error = np.std(np.abs(estimated_counts-real_counts))
print(np.mean(estimated_counts),np.mean(real_counts))
print(mean_abs_error,std_abs_error)
print('image prediction accuracy-->'+str((np.mean(real_counts)-mean_abs_error)/np.mean(real_counts))) 	

