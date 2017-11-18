from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from keras.optimizers import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt
import data_loader as dl
import cnn_generator as cg
import helper_functions as hf
import tensorflow as tf
import scipy.misc as misc
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
import keras.backend as K
import random
import os
import glob
import sys
import argparse

# def get_session(gpu_fraction=1.0):
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
	parser.add_argument("--ku", type=int, default=3)
	parser.add_argument("--img_size", type=int, default=256)
	parser.add_argument("--input_size", type=int, default=100)
	parser.add_argument("--nb_tr_sample", type=int, default=100)
	parser.add_argument("--val_version", type=int, default=15)
	parser.add_argument("--gpu", type=str, default='0')
	args = parser.parse_args()
	return args

args = get_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# stage = 1
# experiment = 'experiment-11.14'
# ku = 3
# img_size = 256
# input_size = 100
# nb_tr_sample =100
# val_version = 15

stage = args.stage
experiment = args.experiment
ku = args.ku
img_size = args.img_size
input_size = args.input_size
nb_tr_sample = args.nb_tr_sample
val_version = args.val_version

# print(val_version)

# if len(sys.argv)>1:
# 	print ("This is the name of the script: ", sys.argv[0])
# 	print ("Number of arguments: ", len(sys.argv))
# 	print ("The arguments are: " , str(sys.argv))
# 	stage = 1
# 	ku = 3
# 	experiment = 'experiment-11.14'

# load data
X_train,X_val,Y_train,Y_val=hf.load_random_data(val_version = val_version)

if stage==0:
	hf.generate_folder(experiment)
	stage_folder = experiment+'/'+'stage-0/'
	hf.generate_folder(stage_folder)
	train_pool_file = stage_folder+'train_pool.txt'
	unnot_pool_file = stage_folder+'unnot_pool.txt'
	train_index_list = []
	index_list = random.sample(range(nb_tr_sample),nb_tr_sample)
	train_index_list = random.sample(range(nb_tr_sample),ku)
	unnot_index_list = [x for x in index_list if x not in train_index_list]
	print(train_pool_file)
	print(unnot_pool_file)
	np.savetxt(train_pool_file, train_index_list)
	np.savetxt(unnot_pool_file, unnot_index_list)

elif stage >= 1:
	stage_folder = experiment+'/'+'stage-'+str(stage)+'/'
	hf.generate_folder(stage_folder)
	pre_stage_folder = experiment+'/'+'stage-'+str(stage-1)+'/'
	model_folders = glob.glob(pre_stage_folder+'round*')
	if len(model_folders)>0:		
		## load the unlabel pool and train pool
		unnot_index_list = []
		train_index_list =[]
		train_pool_file = pre_stage_folder+'train_pool.txt'
		unnot_pool_file = pre_stage_folder+'unnot_pool.txt'
		f = open(train_pool_file)
		for line in f.readlines():
			train_index_list.append(int(float(line.split('\n')[0])))

		f = open(unnot_pool_file)
		unnot_index_list =[]
		for line in f.readlines():
			unnot_index_list.append(int(float(line.split('\n')[0])))

		## load the data
		X_ls = []
		for idx in unnot_index_list:
			X_ls.append(X_train[idx])

		X_un = np.array(X_ls)

		image_patches = hf.image_depatch(X_un, input_size)
		# density_patches = hf.image_depatch(Y_val, 100)
		shp = image_patches.shape
		image_patch_arr = image_patches.reshape(shp[0],shp[1],shp[2],1)
		# density_patch_arr = density_patches.reshape(shp[0],shp[1],shp[2],1)
		all_map_ls = []
		# predictions
		for folder in model_folders:
			K.clear_session()
			model_folder = folder+'/model.h5'
			print(model_folder)
			model = load_model(model_folder)
			preds = model.predict(image_patch_arr).reshape(shp[0],shp[1],shp[2])
			estimated_maps = hf.image_merge(preds,img_size)
			all_map_ls.append(estimated_maps)
	
		all_map_arr = np.array(all_map_ls)
		rot_all_map = np.transpose(all_map_arr, (1,2,3,0))
		pixel_uncertainty_arr = np.var(rot_all_map, axis =3)
		image_uncertainty_arr = np.apply_over_axes(np.sum, pixel_uncertainty_arr, [1,2]).reshape(pixel_uncertainty_arr.shape[0])
		uncertainty_list = image_uncertainty_arr.tolist()
		uncert_ls =[]
		for i in range(len(uncertainty_list)):
			uncert = [uncertainty_list[i], unnot_index_list[i]]
			uncert_ls.append(uncert)

		uncert_ls.sort()
		uncert_ls.reverse()

		selectedSamples = []
		for i in range(ku):
			selectedSamples.append(uncert_ls[i][1])

		train_index_list += selectedSamples
		unnot_list = [x for x in unnot_index_list if x not in selectedSamples]
		## save the train pool and unnotate pool file
		train_pool_file = stage_folder+'train_pool.txt'
		unnot_pool_file = stage_folder+'unnot_pool.txt'
		np.savetxt(train_pool_file, train_index_list)
		np.savetxt(unnot_pool_file, unnot_list)

		## random strategy
		unnot_index_list = []
		train_index_list =[]
		train_pool_file = pre_stage_folder+'train_pool.txt'
		unnot_pool_file = pre_stage_folder+'unnot_pool.txt'
		f = open(train_pool_file)
		for line in f.readlines():
			train_index_list.append(int(float(line.split('\n')[0])))

		f = open(unnot_pool_file)
		unnot_index_list =[]
		for line in f.readlines():
			unnot_index_list.append(int(float(line.split('\n')[0])))

		select_index_list = random.sample(unnot_index_list,ku)
	# 	train_index_list = random.sample(range(nb_tr_sample),ku)
		train_index_list += select_index_list
		train_pool_file = stage_folder+'random_train_pool.txt'
		unnot_pool_file = stage_folder+'random_unnot_pool.txt'
		unnot_index_list = [x for x in unnot_index_list if x not in select_index_list]
		np.savetxt(train_pool_file, train_index_list)
		np.savetxt(unnot_pool_file, unnot_index_list)
	
K.clear_session()
	