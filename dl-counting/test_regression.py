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
# model_path = '9.26-lr-0.001-scaled-batch-100-v9-fcn-200/'+ 'model.h5'
# val_version = 9
# model_path = '9.26-lr-0.0015-scaled-batch-100-v10-fcn-200/'+ 'model.h5'
# model_path = '9.30-lr-0.001-scaled-batch-100-v11-FCN/'+ 'model.h5'
# model_path = '9.30-lr-0.0015-scaled-batch-100-v10-FCN/'+ 'model.h5'

# model_path = '9.28-lr-0.001-scaled-batch-100-v11-fcn-200/'+ 'model.h5'
model_path = '11.4-lr-0.005-scaled-batch-200-v11-ResLR/'+ 'model.h5'
# model_path = '10.31-lr-0.005-scaled-batch-80-v11-ResFCN/'+ 'model.h5'
val_version = 11

# model_path = '10.10-lr-0.005-scaled-batch-100-v12-FCN1/'+ 'model.h5'
# val_version = 12
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
real_count_list = []

for i in range(9):
	preds = model.predict(imagelist[i].reshape(1,200,200,1))
# 	preds = preds/100
# 	preds = preds.reshape(200,200)
	preds_list.append(preds)

# image display
preds_arr = np.array(preds_list)

density_map = hf.patch_merge_for_display(preds_arr, 400)
pred_count = np.sum(density_map)
real_count = np.sum(Images[0,:,:,1])
print([pred_count, real_count])
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1,3,1)
cax=ax.imshow(Images[0,:,:,0])
fig.colorbar(cax)
ax.set_title('Cell image (400x400)')
ax = fig.add_subplot(1,3,2)
cax=ax.imshow(density_map)
fig.colorbar(cax)
ax.set_title('Estimated density (400x400)')
ax.set_xlabel('Cell count:'+str(pred_count))
ax = fig.add_subplot(1,3,3)
cax=ax.imshow(Images[0,:,:,1])
fig.colorbar(cax)
ax.set_title('Ground truth (400x400)')
ax.set_xlabel('Cell count:'+str(real_count))

fig = plt.figure()
pred_count = np.sum(preds_arr[0,:,:])
real_count = np.sum(densitylist[0])
print([pred_count, real_count])
ax = fig.add_subplot(1,3,1)
cax=ax.imshow(imagelist[0].reshape((200,200)))
fig.colorbar(cax)
ax.set_title('Cell image(200x200)')
ax = fig.add_subplot(1,3,2)
cax=ax.imshow(preds_arr[0,:,:])
fig.colorbar(cax)
ax.set_title('Estimated density(200x200)')
ax.set_xlabel('Cell count:'+str(pred_count))
ax = fig.add_subplot(1,3,3)
cax=ax.imshow(densitylist[0].reshape((200,200)))
fig.colorbar(cax)
ax.set_title('Ground truth(200x200)')
ax.set_xlabel('Cell count:'+str(real_count))

# for each image patch
preds_list = []
preds_count_list = []
real_count_list =[]
for i in range(len(imagelist)):
	preds = model.predict(imagelist[i].reshape(1,200,200,1))
# 	preds = preds/100
# 	preds = preds.reshape(200,200)
	preds_list.append(preds)
	preds_count_list.append(np.sum(preds))
	real_count_list.append(np.sum(densitylist[i]))

# fig = plt.figure()
# for i in range(len(imagelist)):
# 	plt.clf()
# 	pred_count = preds_count_list[i]
# 	real_count = real_count_list[i]
# 	print([pred_count, real_count])
# 	ax = fig.add_subplot(1,3,1)
# 	cax = ax.imshow(imagelist[i].reshape((200,200)))
# 	fig.colorbar(cax)
# 	ax.set_title('Cell image(200x200)')
# 	ax = fig.add_subplot(1,3,2)
# 	cax = ax.imshow(preds_list[i])
# 	fig.colorbar(cax)
# 	ax.set_title('Estimated density(200x200)')
# 	ax.set_xlabel('Cell count:'+str(pred_count))
# 	ax = fig.add_subplot(1,3,3)
# 	cax = ax.imshow(densitylist[i].reshape((200,200)))
# 	fig.colorbar(cax)
# 	ax.set_title('Ground truth(200x200)')
# 	ax.set_xlabel('Cell count:'+str(real_count))
# 	plt.pause(0.5)

hf.plot_cell_counts(model.name, preds_count_list, real_count_list)
hf.plot_cell_counts_comp(model.name, preds_count_list, preds_count_list, real_count_list)
# error statistics
abs_err_list = np.abs((np.array(preds_count_list)-np.array(real_count_list)))
mean_abs_err = np.mean(abs_err_list)
std_abs_err = np.std(abs_err_list)
print((mean_abs_err, std_abs_err))


################################# Estimate the density maps based on the real cell images ########################

folder = './minn/'
## load the original cell images
real_path = folder + 'real.dat'
data_num = 26*4*400*400
real_data = np.fromfile(real_path, dtype = np.float, count = data_num, sep ='')
real_data = real_data.reshape(26,4,400,400)
shp = real_data.shape

real_images_list =[]

# decompose the image into image patches for estimation
patches_list =[]
for i in range(shp[0]):
	for j in range(shp[1]):
		image = real_data[i,j,:,:]
		image = 256*(image - np.min(image))/(np.max(image)-np.min(image))
# 		image = image*(image>25)
		real_images_list.append(image)
		patches_list.append(image[:200,:200])
		patches_list.append(image[:200,200:400])
		patches_list.append(image[200:400,:200])
		patches_list.append(image[200:400,200:400])

real_images = np.array(real_images_list).reshape(26,4,400,400)

# estimate the density for each image patch	
preds_list =[]
preds_count_list =[]
for i in range(len(patches_list)):
	preds = model.predict(patches_list[i].reshape(1,200,200,1))
	preds = preds/100
	preds = preds.reshape(200,200)
	preds_list.append(preds)
	preds_count_list.append(np.sum(preds))

# compose the patch density into full image density	
density_maps_list =[]
count_list =[]
for i in range(len(preds_list)):
	if i%4 ==0:
		density_map = np.zeros((shp[2],shp[3]))
		density_map[:200,:200] = preds_list[i]
		density_map[:200,200:400] =preds_list[i+1]
		density_map[200:400,:200] =preds_list[i+2]
		density_map[200:400,200:400] = preds_list[i+3]
		density_maps_list.append(density_map)
		count_list.append(np.sum(np.array(preds_count_list)[i:i+4]))

density_arr = np.reshape(np.array(density_maps_list),shp)
count_arr = np.array(count_list).reshape(-1,4)

## load segmented images
seg_path = folder +'segment.dat'
seg_data = np.fromfile(seg_path, dtype = np.float, count = data_num, sep ='')
seg_data = seg_data.reshape(26,4,400,400)

## load the predicted cell counts from Minn
import glob
import csv
fileNames = glob.glob(folder+'*.csv')

files =[[],[],[],[]]
for f in fileNames:
	if 'cdx2' in f.lower():
		files[0] = f

for f in fileNames:
	if 'dapi' in f.lower():
		files[1] = f
		
for f in fileNames:
	if 'sox17' in f.lower():
		files[2] = f

for f in fileNames:
	if 'sox2' in f.lower():
		files[3] = f

results_list =[]
for file in files:
	f = open(file,'rt')
	reader = csv.reader(f)
	for row in reader:
		results_list.append(float(row[1]))
		
result_arr = np.array(results_list).reshape(shp[1],shp[0])
result_arr = np.transpose(result_arr)

# for row in reader:
# 	print(row)

plt.ion()
fig =plt.figure()

for i in range(shp[0]):
	plt.clf()
	ax =fig.add_subplot(3,4,1)
# 	cax = ax.imshow(real_data[i,1,:,:])
	cax = ax.imshow(real_images[i,1,:,:])
	fig.colorbar(cax)
	ax.set_ylabel('Real cell image')
	ax.set_title('dapi')
	ax =fig.add_subplot(3,4,2)
# 	ax.imshow(real_data[i,0,:,:])
	ax.imshow(real_images[i,0,:,:])
# 	cax = ax.imshow(Images[0,:,:,0])
	fig.colorbar(cax)
	ax.set_title('cxd2')
	ax =fig.add_subplot(3,4,3)
# 	cax = ax.imshow(real_data[i,2,:,:])
	cax = ax.imshow(real_images[i,2,:,:])
	fig.colorbar(cax)
	ax.set_title('sox17')
	ax =fig.add_subplot(3,4,4)
# 	cax = ax.imshow(real_data[i,3,:,:])
	cax = ax.imshow(real_images[i,3,:,:])
	fig.colorbar(cax)
	ax.set_title('sox2')
	ax =fig.add_subplot(3,4,5)
	cax = ax.imshow(seg_data[i,1,:,:])
	fig.colorbar(cax)
	ax.set_ylabel('Minn segmented result')
	ax.set_xlabel('Cell count: '+str(result_arr[i,1]))
	ax =fig.add_subplot(3,4,6)
	cax =  ax.imshow(seg_data[i,0,:,:])
	fig.colorbar(cax)
	ax.set_xlabel('Cell count: '+str(result_arr[i,0]))
	ax =fig.add_subplot(3,4,7)
	cax = ax.imshow(seg_data[i,2,:,:])
	fig.colorbar(cax)
	ax.set_xlabel('Cell count: '+str(result_arr[i,2]))
	ax =fig.add_subplot(3,4,8)
	cax = ax.imshow(seg_data[i,3,:,:])
	fig.colorbar(cax)
	ax.set_xlabel('Cell count: '+str(result_arr[i,3]))
	ax =fig.add_subplot(3,4,9)
	cax = ax.imshow(density_arr[i,1,:,:])
	fig.colorbar(cax)
	ax.set_ylabel('Estimated density map')
	ax.set_xlabel('Cell count: '+str(count_arr[i,1]))
	ax =fig.add_subplot(3,4,10)
	cax =  ax.imshow(density_arr[i,0,:,:])
	fig.colorbar(cax)
	ax.set_xlabel('Cell count: '+str(count_arr[i,0]))
	ax =fig.add_subplot(3,4,11)
	cax = ax.imshow(density_arr[i,2,:,:])
	fig.colorbar(cax)
	ax.set_xlabel('Cell count: '+str(count_arr[i,2]))
	ax =fig.add_subplot(3,4,12)
	cax = ax.imshow(density_arr[i,3,:,:])
	fig.colorbar(cax)
	ax.set_xlabel('Cell count: '+str(count_arr[i,3]))
	plt.pause(1)

## compare the synthetic patch and the real patch
# plt.ion()
# fig = plt.figure()
# plt.clf()
# ax = fig.add_subplot(1,5,1)
# cax = ax.imshow(imagelist[100][:,:,0])
# ax.set_title('Synthetic image patch')
# fig.colorbar(cax)
# ax = fig.add_subplot(1,5,3)
# cax = ax.imshow(patches_list[1])
# ax.set_title('Real patch (cdx2)')
# fig.colorbar(cax)
# ax = fig.add_subplot(1,5,2)
# cax = ax.imshow(patches_list[5])
# ax.set_title('Real patch (dapi)')
# fig.colorbar(cax)
# ax = fig.add_subplot(1,5,4)
# cax = ax.imshow(patches_list[9])
# ax.set_title('Real patch (sox17)')
# fig.colorbar(cax)
# ax = fig.add_subplot(1,5,5)
# cax = ax.imshow(patches_list[13])
# ax.set_title('Real patch (sox2)')
# fig.colorbar(cax)

# read train_loss and test_loss txt file
# import glob
# folder = '9.26-lr-0.0015-scaled-batch-100-v10-fcn-200/'
# txtfiles = glob.glob(folder +'*.txt')
# 
# tr_loss =[]
# val_loss =[]
# loss_list =[]
# for i in range(len(txtfiles)):
# 	t = txtfiles[i]
# 	f = open(t)
# 	for line in f.readlines():
# 		loss_list.append(float(line.rstrip()))
# 
# tr_loss = loss_list[int(len(loss_list)/2):]
# val_loss = loss_list[:int(len(loss_list)/2)]
# 
# hf.plot_loss(folder, tr_loss, val_loss)