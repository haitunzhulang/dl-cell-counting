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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
input_shape=[(100,100)]


# load the model
# model_path = 'lr-0.001-scaled-batch-480/'+ 'model.h5'
# model_path = '9.26-lr-0.001-scaled-batch-100-v9-fcn-200/'+ 'model.h5'
# val_version = 9
# model_path = '9.26-lr-0.0015-scaled-batch-100-v10-fcn-200/'+ 'model.h5'
# model_path = '9.30-lr-0.001-scaled-batch-100-v11-FCN/'+ 'model.h5'
# model_path = '9.30-lr-0.0015-scaled-batch-100-v10-FCN/'+ 'model.h5'

# model_path = '9.28-lr-0.001-scaled-batch-100-v11-fcn-200/'+ 'model.h5'
# model_path = '11.4-lr-0.005-scaled-batch-100-v15-ResFCN100/'+ 'model1.h5'
# model_path = '11.5-lr-0.01-scaled-batch-95-v15-ResFCN100/'+ 'model.h5'
# model_path = '11.7-lr-0.01-scaled-batch-180-v15-ResFCN100/'+ 'model.h5'
# model_path = '11.7-lr-0.01-scaled-batch-180-v15-ResFCN100/dt-11.8-lr-0.0001-scaled-batch-150-v14-ResFCN-400-normalized/'+ 'model.h5'
# model_path = '11.7-lr-0.01-scaled-batch-180-v15-ResFCN100/dt-11.9-lr-5e-06-scaled-batch-150-v14-ResFCN-400/'+ 'model.h5'
# model_path = '11.8-lr-0.01-scaled-batch-170-v15-ResFCN100/stage-1-round-4dt-11.13-lr-5e-06-scaled-batch-150-v14-ResFCN-400/'+ 'model.h5'
model_path = '11.7-lr-0.01-scaled-batch-180-v15-ResFCN100/stage-1-round-1dt-11.13-lr-5e-06-scaled-batch-150-v14-ResFCN-400/'+ 'model.h5'
model=load_model(model_path)


val_version = 14
train_include =1
image_size = 512
# model_path = '10.10-lr-0.005-scaled-batch-100-v12-FCN1/'+ 'model.h5'
# val_version = 12
	
X_train,X_val,Y_train,Y_val=hf.load_random_data(val_version = val_version)
if train_include == 1:
	X_val = np.concatenate([X_train, X_val], axis =0)
	Y_val = np.concatenate([Y_train, Y_val], axis =0)

# me_den = np.repeat(np.apply_over_axes(np.mean, X_val, [1,2]).reshape(-1,1),X_train.shape[1]*X_train.shape[1], axis =1).reshape(X_val.shape[0],X_train.shape[1],X_train.shape[1])
# std_den = np.repeat(np.apply_over_axes(np.std, X_val, [1,2]).reshape(-1,1),X_train.shape[1]*X_train.shape[1], axis =1).reshape(X_val.shape[0],X_train.shape[1],X_train.shape[1])
# X_val_norm = (X_val - me_den)/std_den
# image_patches = hf.image_depatch(X_val_norm, 100)
image_patches = hf.image_depatch(X_val, 100)

## normalize each patch
# me_den = np.repeat(np.apply_over_axes(np.mean, image_patches, [1,2]).reshape(-1,1),100*100, axis =1).reshape(image_patches.shape[0],100,100)
# std_den = np.repeat(np.apply_over_axes(np.std, image_patches, [1,2]).reshape(-1,1),100*100, axis =1).reshape(image_patches.shape[0],100,100)
# max_den = np.repeat(np.apply_over_axes(np.max, image_patches, [1,2]).reshape(-1,1),100*100, axis =1).reshape(image_patches.shape[0],100,100)
# min_den = np.repeat(np.apply_over_axes(np.min, image_patches, [1,2]).reshape(-1,1),100*100, axis =1).reshape(image_patches.shape[0],100,100)
# image_patches = (image_patches - me_den)/std_den
density_patches = hf.image_depatch(Y_val, 100)

# patch_list = []
# density_list =[]
# for i in range(image_patches.shape[0]):
# 	kth = i%36
# 	xth = kth//6
# 	yth = kth%6
# 	if xth == 0 or xth == 5 or yth == 0 or yth ==5:
# 		continue
# 	else:
# 		patch_list.append(image_patches[i,:])
# 		density_list.append(density_patches[i,:])

# image_patches = np.array(patch_list)
# density_patches = np.array(density_list)

shp = image_patches.shape
image_patch_arr = image_patches.reshape(shp[0],shp[1],shp[2],1)
density_patch_arr = density_patches.reshape(shp[0],shp[1],shp[2],1)


# density map estimation
preds = model.predict(image_patch_arr)/100
preds = preds.reshape(-1,shp[1],shp[2])
pred_counts = np.apply_over_axes(np.sum,preds,[1,2]).reshape(preds.shape[0])
ground_counts = np.apply_over_axes(np.sum,density_patch_arr,[1,2]).reshape(preds.shape[0])
mean_abs_error = np.mean(np.abs(pred_counts-ground_counts))
std_abs_error = np.std(np.abs(pred_counts-ground_counts))
print(np.mean(pred_counts),np.mean(ground_counts))
print(mean_abs_error,std_abs_error)
print('patch prediction accuracy-->'+str((np.mean(ground_counts)-mean_abs_error)/np.mean(ground_counts)))

# merge the image
estimated_maps = hf.image_merge(preds,image_size)
estimated_counts = np.apply_over_axes(np.sum,estimated_maps,[1,2]).reshape(estimated_maps.shape[0])
real_counts = np.apply_over_axes(np.sum,Y_val,[1,2]).reshape(estimated_maps.shape[0])
mean_abs_error = np.mean(np.abs(estimated_counts-real_counts))
std_abs_error = np.std(np.abs(estimated_counts-real_counts))
print(np.mean(estimated_counts),np.mean(real_counts))
print(mean_abs_error,std_abs_error)
print('image prediction accuracy-->'+str((np.mean(real_counts)-mean_abs_error)/np.mean(real_counts)))

#hf.plot_cell_counts_comp(model.name,pred_counts.tolist(), pred_counts.tolist(), ground_counts.tolist())
# hf.plot_cell_counts_comp(model.name,estimated_counts.tolist(), estimated_counts.tolist(), real_counts.tolist())
hf.plot_cell_counts(model.name, estimated_counts.tolist(), real_counts.tolist())

fig = plt.figure()
hf.visualize_results(fig,image_patch_arr,preds,density_patch_arr,image_patch_arr.shape[0],2)
# for i in range(pred_counts.shape[0]):
# 	plt.clf()
# 	pred_count = pred_counts[i]
# 	real_count = ground_counts[i]
# 	print([pred_count, real_count])
# 	ax = fig.add_subplot(1,3,1)
# 	image = image_patches[i,:,:]
# # 	image = image*(image>15)
# 	cax = ax.imshow(image)
# 	fig.colorbar(cax)
# 	ax.set_title('Cell image(100x100)')
# 	ax = fig.add_subplot(1,3,2)
# 	cax = ax.imshow(preds[i,:])
# 	fig.colorbar(cax)
# 	ax.set_title('Estimated density(100x100)')
# 	ax.set_xlabel('Cell count:'+str(pred_count))
# 	ax = fig.add_subplot(1,3,3)
# 	cax = ax.imshow(density_patches[i,:])
# 	fig.colorbar(cax)
# 	ax.set_title('Ground truth(100x100)')
# 	ax.set_xlabel('Cell count:'+str(real_count))
# 	plt.pause(1)

fig = plt.figure()
hf.visualize_results(fig, X_val, estimated_maps, Y_val, X_val.shape[0],2)

## print out the estimation and ground truth
for i in range(len(estimated_counts)):
	print(estimated_counts[i])

for i in range(len(real_counts)):
	print(real_counts[i])
# for i in range(estimated_maps.shape[0]):
# 	plt.clf()
# 	pred_count = estimated_counts[i]
# 	real_count = real_counts[i]
# 	print([pred_count, real_count])
# 	ax = fig.add_subplot(1,3,1)
# 	image = X_val[i,:,:]
# # 	image = image*(image>15)
# 	cax = ax.imshow(image)
# 	fig.colorbar(cax)
# 	ax.set_title('Cell image('+str(image.shape[1])+'x'+str(image.shape[1])+')')
# 	ax = fig.add_subplot(1,3,2)
# 	cax = ax.imshow(estimated_maps[i,:])
# 	fig.colorbar(cax)
# 	ax.set_title('Estimated density('+str(image.shape[1])+'x'+str(image.shape[1])+')')
# 	ax.set_xlabel('Cell count:'+str(pred_count))
# 	ax = fig.add_subplot(1,3,3)
# 	cax = ax.imshow(Y_val[i,:])
# 	fig.colorbar(cax)
# 	ax.set_title('Ground truth('+str(image.shape[1])+'x'+str(image.shape[1])+')')
# 	ax.set_xlabel('Cell count:'+str(real_count))
# 	plt.pause(2)

# Y_val=Y_val.reshape((Y_val.shape[0],Y_val.shape[1],Y_val.shape[2],1))
# X_val=X_val.reshape((X_val.shape[0],X_val.shape[1],X_val.shape[2],1))

# Images=np.concatenate([X_val,Y_val],axis=3)
# batch = Images
# 
# imagelist, densitylist = hf.patch_gen_for_test(batch, 200)

