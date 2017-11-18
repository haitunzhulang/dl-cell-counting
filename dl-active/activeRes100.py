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

# KTF.set_session(get_session())

def get_args():
	parser = argparse.ArgumentParser()
	now = datetime.datetime.now()
	date = now.strftime("%Y.%m.%d")
	parser.add_argument("--stage", type=int, default=0)
	parser.add_argument("--experiment", type=str, default='experiment-11.14')
	parser.add_argument("--input_size", type=int, default=100)
	parser.add_argument("--val_version", type=int, default=15)
	parser.add_argument("--date", type=str, default=date)
	parser.add_argument("--nb_round", type=int, default=4)
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--nb_epoch_per_record", type=int, default=2)
	parser.add_argument("--lr", type=int, default=0.01)
	parser.add_argument("--batch_size", type=int, default=170)
	parser.add_argument("--gpu", type=str, default='0')
	parser.add_argument("--round", type=int, default=0)
	args = parser.parse_args()
	return args

args = get_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
stage = args.stage
experiment = args.experiment
input_size = args.input_size
val_version = args.val_version
nb_epochs = args.nb_epochs
nb_epoch_per_record = args.nb_epoch_per_record
input_shape=[(input_size,input_size)]
date = args.date
stage = args.stage
nb_round = args.nb_round
lr=args.lr
batch_size = args.batch_size
round = args.round
# model=cg.fcn200()
# model=cg.FCN()
# model=cg.FCN1()

#X_train,X_val,Y_train,Y_val=hf.load_random_data()
X_train,X_val,Y_train,Y_val=hf.load_random_data(val_version=val_version)

# training data selection
train_folder = experiment+'/'+'stage-'+str(stage)+'/'
train_pool_file = train_folder+'train_pool.txt'
#unnot_pool_file = train_folder+'unnot_pool.txt'
f = open(train_pool_file)
train_pool_list =[]
for line in f.readlines():
	train_pool_list.append(int(float(line.split('\n')[0])))
X_sl = X_train[train_pool_list,:,:]
Y_sl = Y_train[train_pool_list,:,:]

# batch_size = int(X_train.shape[0]*34/20)
# scale the ground truth by multiplying 100
Y_sl = Y_sl * 100
Y_val = Y_val * 100

# for round in [round]:
print('********** ROUND '+str(round)+'**********')
model = K.clear_session()
model=cg.ResFCN100()
sgd = SGD(lr=lr, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
model_name = train_folder+'round-'+str(round)+'-dt-'+date+'-lr-'+str(lr)+ '-scaled'+'-batch-'+str(batch_size)+'-v'+str(val_version)+'-ResFCN100'
# model_name = '9.30-'+'lr-'+str(lr)+ '-scaled'+'-batch-'+str(batch_size)+'-v'+str(val_version)+'-FCN'
model.name = model_name
hf.train_model(model, X_sl,X_val,Y_sl,Y_val, nb_epochs=nb_epochs, nb_epoch_per_record=nb_epoch_per_record, input_shape=input_shape[0], batch_size = batch_size)

if stage >=1:
	## random strategy
	train_pool_file = train_folder+'random_train_pool.txt'
	f = open(train_pool_file)
	train_pool_list =[]
	for line in f.readlines():
		train_pool_list.append(int(float(line.split('\n')[0])))
	X_sl = X_train[train_pool_list,:,:]
	Y_sl = Y_train[train_pool_list,:,:]

	# scale the ground truth by multiplying 100
	Y_sl = Y_sl * 100
# 	Y_val = Y_val * 100
# 	for round in range(nb_round):
	print('********** Random ROUND '+str(round)+'**********')
	model = K.clear_session()
	model=cg.ResFCN100()
	sgd = SGD(lr=lr, decay=1e-4, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer=sgd)
	model_name = train_folder+'random_round-'+str(round)+'-dt-'+date+'-lr-'+str(lr)+ '-scaled'+'-batch-'+str(batch_size)+'-v'+str(val_version)+'-ResFCN100'
	# model_name = '9.30-'+'lr-'+str(lr)+ '-scaled'+'-batch-'+str(batch_size)+'-v'+str(val_version)+'-FCN'
	model.name = model_name
	hf.train_model(model, X_sl,X_val,Y_sl,Y_val, nb_epochs=nb_epochs, nb_epoch_per_record=nb_epoch_per_record, input_shape=input_shape[0], batch_size = batch_size)

## for synchronization
sync_file = train_folder+'sync-'+str(round)
with open(sync_file,'w') as syn_f:
	syn_f.write('OK')
K.clear_session()