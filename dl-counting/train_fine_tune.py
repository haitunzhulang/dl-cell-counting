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


# load the model
model_folder = '10.31-lr-0.005-scaled-batch-80-v11-ResFCN/'
model_path = model_folder+ 'model.h5'
model=load_model(model_path)

val_version = 14
nb_epochs = 10000
nb_epoch_per_record = 2
input_shape = [(200,200)]
batch_size = 60
lr=0.00001
date = '11.3'
threshold = 1
# load data
X_train, X_val, Y_train, Y_val=hf.load_random_data(val_version = val_version)
if threshold == 1:
	X_train = (X_train>15)* X_train
	X_val = (X_val>15)* X_val
	X_train = X_train[:,256-200:256+200, 256-200:256+200]
	X_val = X_val[:,256-200:256+200, 256-200:256+200]

sgd = SGD(lr=lr, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model_name = model_folder + 'dt-'+date+'-lr-'+str(lr)+ '-scaled'+'-batch-'+str(batch_size)+'-v'+str(val_version)+'-ResFCN-'+str(X_train.shape[1])
# model_name = '9.30-'+'lr-'+str(lr)+ '-scaled'+'-batch-'+str(batch_size)+'-v'+str(val_version)+'-FCN'
model.name = model_name
hf.train_model(model, X_train,X_val,Y_train,Y_val, nb_epochs=nb_epochs, nb_epoch_per_record=nb_epoch_per_record, input_shape=input_shape[0], batch_size = batch_size)