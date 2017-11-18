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
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


stage = 1
experiment = 11.2

model_path1 = '10.04-lr-0.0015-scaled-batch-100-v10-fcn200/' + 'model.h5'
model_path2 = '10.11-lr-0.005-scaled-batch-100-v11-FCN/' + 'model.h5'
model_path3 = '10.09-lr-0.005-scaled-batch-100-v11-fcn200/' + 'model.h5'
model_path4 = '9.19-lr-0.001-scaled-batch-100v-3fcn-200/' + 'model.h5'

model_paths = [model_path1, model_path2, model_path3, model_path4]

val_version = 11
# model=load_model(model_path)	
# lr=0.0005
# model=cg.fcn32()
# sgd = SGD(lr=lr, decay=1e-4, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer=sgd)
X_train,X_val,Y_train,Y_val=hf.load_random_data(val_version = val_version)
# Y_train=Y_train.reshape((Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))
# X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
# Images=np.concatenate([X_train,Y_train],axis=3)
Y_val=Y_val.reshape((Y_val.shape[0],Y_val.shape[1],Y_val.shape[2],1))
X_val=X_val.reshape((X_val.shape[0],X_val.shape[1],X_val.shape[2],1))
Images=np.concatenate([X_val,Y_val],axis=3)
batch = Images
imagelist, densitylist = hf.patch_gen_for_test(batch, 200)

feature_list =[]
prediction_list =[]
feat_list =[]
pred_list =[]
for model_path in model_paths:
	model = load_model(model_path)
	layer_outputs = [layer.output for layer in model.layers]
	viz_model = Model(input=model.input, output=layer_outputs)
	for image in imagelist:
		x = image.reshape(1,200,200,1)
		features = viz_model.predict(x)
		feat_list.append(features[23])
		pred_list.append(features[-1])

## reorganize the data
# feat_ls = []
# pred_ls =[]
# for i in range(len(pred_list)):
# 	feat_ls.append(feat_list[i].reshape(200,200))
# 	pred_ls.append(pred_list[i].reshape(200,200))

ln = len(imagelist)
feature_list.append(feat_list[:ln])
feature_list.append(feat_list[ln:ln*2])
feature_list.append(feat_list[ln*2:ln*3])
feature_list.append(feat_list[ln*3:ln*4])
prediction_list.append(pred_list[:ln])
prediction_list.append(pred_list[ln:ln*2])
prediction_list.append(pred_list[ln*2:ln*3])
prediction_list.append(pred_list[ln*3:ln*4])
feat_arr = np.array(feature_list).reshape(4,900,24,24,512)
pred_arr = np.array(prediction_list).reshape(4,900,200,200)


## top K uncertain samples
Ku = 30
ku =20

## calculate the uncertainty
pred_arr_rot = np.transpose(pred_arr, (1,2,3,0))
pixel_uncertainty_arr = np.var(pred_arr_rot, axis =3)
patch_uncertainty_arr = np.apply_over_axes(np.sum, pixel_uncertainty_arr, [1,2]).reshape(100,9)
image_uncertainty_arr = np.mean(patch_uncertainty_arr, axis =1)
uncertainty_list = image_uncertainty_arr.tolist()
uncert_ls =[]
for i in range(len(uncertainty_list)):
	uncert = [uncertainty_list[i], i]
	uncert_ls.append(uncert)
uncert_ls.sort()
uncert_ls.reverse()

## calculate the similarity
ch_wise_feat_arr = np.apply_over_axes(np.sum, feat_arr, [2,3]).reshape(4,900,512)
feat_arr_rot = np.transpose(ch_wise_feat_arr, (1,0,2)).reshape(900,512*4)

from scipy import spatial
sim_ls =[]
for i in range(900):
	for sp in range(900):
		sim_ls.append(1-spatial.distance.cosine(feat_arr_rot[i], feat_arr_rot[sp]))
sim_arr = np.array(sim_ls).reshape(900,900)
sim_vec = np.sum(sim_arr,1)/900
sim_vec_ls = sim_vec.tolist()
sim_vec_ls.sort()



## annotation suggestion
unlabelPool = []
candidatePool = []
selectedSamples = []
trainPool =[]
nb_sample = int(ln/9)
unlabelPool = range(nb_sample)
sim_list = []
for i in range(Ku):
	candidatePool.append(uncert_ls[i][1])
	
from scipy import spatial
for i in range(Ku):
	idx = candidatePool[i]
	for sp in unlabelPool:
		sim_list.append(1-spatial.distance.cosine(feat_arr_rot[idx], feat_arr_rot[sp]))

sim_arr = np.array(sim_list).reshape(30,100)
sim_vec = np.sum(sim_arr,1)

trainPool += candidatePool

from scipy import spatial
sim_ls =[]
for i in range(100):
	for sp in unlabelPool:
		sim_ls.append(1-spatial.distance.cosine(feat_arr_rot[i], feat_arr_rot[sp]))
sim_arr = np.array(sim_ls).reshape(100,100)
sim_vec = np.sum(sim_arr,1)
sim_vec_ls = sim_vec.tolist()
sim_vec_ls.sort()
	
preds_list = []
preds_count_list = []
real_count_list = []

x = imagelist[0].reshape(1,200,200,1)
features = viz_model.predict(x)



for i in range(9):
	preds = model.predict(imagelist[i].reshape(1,200,200,1))
	preds = preds/100
	preds = preds.reshape(200,200)
	preds_list.append(preds)

nb_models = 4
date = 10.31

unlabelPool = []
trainPool =[]

#### load the data ####


#### model training ####
for i in range(nb_models):
	os.system()

#### metric computing ####
# ---- uncertainty ----


# ---- similarity ----


#### annotation suggestion ####
