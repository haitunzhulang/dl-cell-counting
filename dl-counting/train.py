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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
val_version=3

def get_session(gpu_fraction=1.0):
#     '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# KTF.set_session(get_session())

nb_epochs = 1000
nb_epoch_per_record =1
input_shape=[(100,100)]
val_version = 3


lr=0.0005
model=cg.fcn32A()
sgd = SGD(lr=lr, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)


#X_train,X_val,Y_train,Y_val=hf.load_random_data()
X_train,X_val,Y_train,Y_val=hf.load_random_data(val_version=val_version)

batch_size = X_train.shape[0]*2
# scale the ground truth by multiplying 100
Y_train = Y_train * 100
Y_val = Y_val * 100

model_name = '9.17-'+'lr-'+str(lr)+ '-scaled'+'-batch-'+str(batch_size)+'v-'+str(val_version)+'fcn-A'
model.name = model_name
hf.train_model(model, X_train,X_val,Y_train,Y_val, nb_epochs=nb_epochs, nb_epoch_per_record=nb_epoch_per_record, input_shape=input_shape[0], batch_size = batch_size)
	