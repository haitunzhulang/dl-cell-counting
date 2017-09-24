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
val_version=3

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

nb_epochs = 100
nb_epoch_per_record =1
input_shape=[(200,200)]

# load the model
# model_path = 'lr-0.001-scaled-batch-480/'+ 'model.h5'
model_path = '9.21-lr-0.001-scaled-batch-100v-3fcn-200/'+ 'model.h5'
val_version = 3
model=load_model(model_path)
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

# plt.ion()
# fig = plt.figure()
preds_list = []
preds_count_list = []
real_count_list =[]
for i in range(9):
	preds = model.predict(imagelist[i].reshape(1,200,200,1))
	preds = preds/100
	preds = preds.reshape(200,200)
	preds_list.append(preds)
	

# image display
preds_arr = np.array(preds_list)

density_map = hf.patch_merge_for_display(preds_arr, 250)
pred_count = np.sum(density_map)
real_count = np.sum(Images[0,:,:,1])
print([pred_count, real_count])
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1,3,1)
ax.imshow(Images[0,:,:,0])
ax.set_title('Cell image (250x250)')
ax = fig.add_subplot(1,3,2)
ax.imshow(density_map)
ax.set_title('Estimated density (250x250)')
ax.set_xlabel('Cell count:'+str(pred_count))
ax = fig.add_subplot(1,3,3)
ax.imshow(Images[0,:,:,1])
ax.set_title('Ground truth (250x250)')
ax.set_xlabel('Cell count:'+str(real_count))

fig = plt.figure()
pred_count = np.sum(preds_arr[0,:,:])
real_count = np.sum(densitylist[0])
print([pred_count, real_count])
ax = fig.add_subplot(1,3,1)
ax.imshow(imagelist[0].reshape((200,200)))
ax.set_title('Cell image(200x200)')
ax = fig.add_subplot(1,3,2)
ax.imshow(preds_arr[0,:,:])
ax.set_title('Estimated density(200x200)')
ax.set_xlabel('Cell count:'+str(pred_count))
ax = fig.add_subplot(1,3,3)
ax.imshow(densitylist[0].reshape((200,200)))
ax.set_title('Ground truth(200x200)')
ax.set_xlabel('Cell count:'+str(real_count))

# for each image patch
preds_list = []
preds_count_list = []
real_count_list =[]
for i in range(len(imagelist)):
	preds = model.predict(imagelist[i].reshape(1,200,200,1))
	preds = preds/100
	preds = preds.reshape(200,200)
	preds_list.append(preds)
	preds_count_list.append(np.sum(preds))
	real_count_list.append(np.sum(densitylist[i]))

# fig = plt.figure()
# for i in range(len(imagelist)):
# 	pred_count = preds_count_list[i]
# 	real_count = real_count_list[i]
# 	print([pred_count, real_count])
# 	ax = fig.add_subplot(1,3,1)
# 	ax.imshow(imagelist[i].reshape((200,200)))
# 	ax.set_title('Cell image(200x200)')
# 	ax = fig.add_subplot(1,3,2)
# 	ax.imshow(preds_list[i])
# 	ax.set_title('Estimated density(200x200)')
# 	ax.set_xlabel('Cell count:'+str(pred_count))
# 	ax = fig.add_subplot(1,3,3)
# 	ax.imshow(densitylist[i].reshape((200,200)))
# 	ax.set_title('Ground truth(200x200)')
# 	ax.set_xlabel('Cell count:'+str(real_count))
# 	plt.pause(0.5)

hf.plot_cell_counts(model.name, preds_count_list, real_count_list)
# error statistics
abs_err_list = np.abs((np.array(preds_count_list)-np.array(real_count_list)))
mean_abs_err = np.mean(abs_err_list)
std_abs_err = np.std(abs_err_list)


train_path = './real.dat'
data_num = 26*4*400*400
train_data = np.fromfile(train_path,dtype=np.float,count=data_num,sep='')
train_data = train_data.reshape(26,4,400,400)
shp = train_data.shape

result_list=[]
for i in range(shp[1]):
	count_list =[]
	for j in range(shp[0]):
		image = train_data[j,i,:,:]
		image = 256*(train_data - np.min(train_data))/(np.max(train_data)-np.min(train_data))
		imagelist = []
		imagelist.append(image[:200,:200])
		imagelist.append(image[:200,200:400])
		imagelist.append(image[200:400,:200])
		imagelist.append(image[200:400,200:400])
		preds_list = []
		preds_count_list = []
		for i in range(len(imagelist)):
			preds = model.predict(imagelist[i].reshape(1,200,200,1))
			preds = preds/100
			preds = preds.reshape(200,200)
			preds_list.append(preds)
			preds_count_list.append(np.sum(preds))
		count_list.append(sum(preds_count_list))
	result_list.append(count_list)


# plt.ion()
# fig = plt.figure()
# for i in range(len(imagelist)):
# 	preds = model.predict(imagelist[i].reshape(1,100,100,1))
# 	preds = preds/100
# 	pred_count = np.sum(preds)
# 	real_count = np.sum(densitylist[i])
# 	print([pred_count, real_count])
# 	ax = fig.add_subplot(1,3,1)
# 	ax.imshow(imagelist[i].reshape((100,100)))
# 	ax.set_title('Cell image')
# 	ax = fig.add_subplot(1,3,2)
# 	ax.imshow(preds.reshape((100,100)))
# 	ax.set_title('Estimated density')
# 	ax = fig.add_subplot(1,3,3)
# 	ax.imshow(densitylist[i].reshape((100,100)))
# 	ax.set_title('Ground truth')
# 	plt.pause(1)


