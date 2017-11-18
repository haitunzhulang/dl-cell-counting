import numpy as np
import matplotlib.pyplot as plt
import data_loader as dl
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from math import floor
import os



folder = '/home/shenghua/dl-cell-counting/mt-cell-train-data/dataset/'

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)
        
# plot and save the file
def plot_save(model_name,loss,val_loss,acc,val_acc):
    generate_folder(model_name)
    f_out=model_name+'/loss_acc_kappa.png'
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8,4))
    ax = fig.add_subplot(1,2,1)
    ax.plot(loss,'b-',linewidth=1.3)
    ax.plot(val_loss,'r-',linewidth=1.3)
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('epoches')
    ax.legend(['train', 'test'], loc='upper left')
    ax = fig.add_subplot(1,2,2)
    ax.plot(acc,'b-', linewidth=1.3)
    ax.plot(val_acc,'r-',linewidth=1.3)
    ax.set_title('Model Acc')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('epoches')
    ax.legend(['train acc', 'test acc'], loc='upper left')   
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(f_out, dpi=80)
    
# plot and save the file
def plot_loss(model_name,loss,val_loss):
    generate_folder(model_name)
    f_out=model_name+'/loss_epochs.png'
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(loss,'b-',linewidth=1.3)
    ax.plot(val_loss,'r-',linewidth=1.3)
    ax.set_title('Model Loss')
    ax.set_ylabel('MSE')
    ax.set_xlabel('epochs')
    ax.legend(['train', 'test'], loc='upper left')  
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(f_out, dpi=80)
    
# plot mean absolute error and std
def plot_mae(model_name,mae_list,std_list):
    generate_folder(model_name)
    f_out=model_name+'/mean_absolute_error.png'
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(mae_list,'b-',linewidth=1.3)
    ax.plot(std_list,'r-',linewidth=1.3)
    ax.set_title('Stats of Absolute error')
    ax.set_ylabel('Error')
    ax.set_xlabel('epochs')
    ax.legend(['MAE', 'STD'], loc='upper left')  
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(f_out, dpi=80)

# plot and save the file
def plot_cell_counts(model_name,preds_count_list,real_count_list):
    generate_folder(model_name)
    f_out=model_name+'/cell_count.png'
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(preds_count_list,'b-',linewidth=1.3)
    ax.plot(real_count_list,'r-',linewidth=1.3)
    ax.set_title('Cell counts')
    ax.set_ylabel('Count of cells')
    ax.set_xlabel('Patchs')
    ax.legend(['Prediction', 'Ground truth'], loc='upper left')  
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(f_out, dpi=80)
    
def plot_cell_counts_comp(model_name,preds_count_list, preds_count_list2,real_count_list):
	generate_folder(model_name)
	f_out=model_name+'/cell_count_comp.png'
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	# sort the results by ordering the cell counts in the ground truth images
	count_list = np.array([real_count_list, preds_count_list, preds_count_list2]).transpose().tolist()
	count_list.sort()
	count_list_sorted = np.array(count_list).transpose().tolist()
	real_count_list = count_list_sorted[0]
	print(np.mean(real_count_list[0:500]))
	preds_count_list = count_list_sorted[1]
	preds_count_list2 = count_list_sorted[2]
	abs_err_list = np.abs(np.array(preds_count_list[0:500])-np.array(real_count_list[0:500]))
	mean_abs_err = np.mean(abs_err_list)
	std_abs_err = np.std(abs_err_list)
	print([mean_abs_err, std_abs_err])
	abs_err_list = np.abs(np.array(preds_count_list2[0:500])-np.array(real_count_list[0:500]))
	mean_abs_err = np.mean(abs_err_list)
	std_abs_err = np.std(abs_err_list)
	print([mean_abs_err, std_abs_err])
	fig = Figure(figsize=(12,6))
	ax = fig.add_subplot(1,1,1)
	ax.plot(preds_count_list,'b-',linewidth=1.0)
	ax.plot(preds_count_list2,'g-',linewidth=0.6)
	ax.plot(real_count_list,'r-',linewidth=1.0)
	ax.set_title('Cell counts')
	ax.set_ylabel('Count of cells')
	ax.set_xlabel('Patchs')
	ax.legend(['FCRN estimation', 'FCN estimation', 'Ground truth'], loc='upper left')
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(f_out, dpi=100)

def load_random_data(val_version=1):
	imageFileName = folder + 'imageSet.dat'
	densityFileName = folder + 'densitySet.dat'
	if val_version==1:
		# train data dir
		print('validation version:'+str(val_version))
		data=dl.train_load(imageFileName);
		truthImages=dl.truth_load(densityFileName);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data=data[0:240,:,:]
		test_data=data[240:300,:,:]
		train_truths=truthImages[0:240,:,:]
		test_truths=truthImages[240:300,:,:]

	if val_version==2:
		# train data dir
		print('validation version:'+str(val_version))
		data=dl.train_load(imageFileName);
		truthImages=dl.truth_load(densityFileName);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data=data[0:150,:,:]
		test_data=data[150:300,:,:]
		train_truths=truthImages[0:150,:,:]
		test_truths=truthImages[150:300,:,:]

	if val_version==3:
		# train data dir
		print('validation version:'+str(val_version))
		data=dl.train_load(imageFileName);
		truthImages=dl.truth_load(densityFileName);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data=data[0:200,:,:]
		test_data=data[200:300,:,:]
		train_truths=truthImages[0:200,:,:]
		test_truths=truthImages[200:300,:,:]

	if val_version==4:
		# 400 x 400 training data, cell size: 9 pixels, 
		# cell number: [200-600], [600-1200], [1200-2000]
		imageFileName = folder + 'imageSet9.dat'
		densityFileName = folder + 'densitySet9.dat'
		print(imageFileName)
		print(densityFileName)
		print('validation version:'+str(val_version))
		image_shape = (400,400)
		image_number = 300
		data=dl.train_data_load(imageFileName, image_shape, image_number);
		truthImages=dl.truth_data_load(densityFileName, image_shape, image_number);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data = data[0:200,:,:]
		test_data = data[200:300,:,:]
		train_truths = truthImages[0:200,:,:]
		test_truths = truthImages[200:300,:,:]

	if val_version==5:
		# 400 x 400 training data, cell size: 9 pixels, density: 2
		# cell number: [200-600], [600-1200], [1200-2000]
		imageFileName = folder + 'imageSet9s.dat'
		densityFileName = folder + 'densitySet9s.dat'
		print(imageFileName)
		print(densityFileName)
		print('validation version:'+str(val_version))
		image_shape = (400,400)
		image_number = 300
		data=dl.train_data_load(imageFileName, image_shape, image_number);
		truthImages=dl.truth_data_load(densityFileName, image_shape, image_number);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data = data[0:200,:,:]
		test_data = data[200:300,:,:]
		train_truths = truthImages[0:200,:,:]
		test_truths = truthImages[200:300,:,:]

	if val_version==6:
		# 400 x 400 training data, cell size: 9 pixels, density: 4
		# cell number: [200-600], [600-1200], [1200-2000]
		imageFileName = folder + 'imageSet9s1.dat'
		densityFileName = folder + 'densitySet9s1.dat'
		print(imageFileName)
		print(densityFileName)
		print('validation version:'+str(val_version))
		image_shape = (400,400)
		image_number = 300
		data=dl.train_data_load(imageFileName, image_shape, image_number);
		truthImages=dl.truth_data_load(densityFileName, image_shape, image_number);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data = data[0:200,:,:]
		test_data = data[200:300,:,:]
		train_truths = truthImages[0:200,:,:]
		test_truths = truthImages[200:300,:,:]

	if val_version==7:
		# 400 x 400 training data, cell size: 8 pixels, kernel: 4.5x5
		# cell number: [200-600], [600-1000], [1000-1500]
		imageFileName = folder + 'imageSet_c8_k4.dat'
		densityFileName = folder + 'densitySet_c8_k4.dat'
		print(imageFileName)
		print(densityFileName)
		print('validation version:'+str(val_version))
		image_shape = (400,400)
		image_number = 300
		data = dl.train_data_load(imageFileName, image_shape, image_number);
		# normalize the data range into [0,255]
		for i in range(data.shape[0]):
			image = data[i,:,:]
			data[i,:,:] = 255*(image - np.min(image))/(np.max(image)-np.min(image))
		truthImages=dl.truth_data_load(densityFileName, image_shape, image_number);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data = data[0:200,:,:]
		test_data = data[200:300,:,:]
		train_truths = truthImages[0:200,:,:]
		test_truths = truthImages[200:300,:,:]

	if val_version==8:
		# 400 x 400 training data, cell size: 8 pixels, kernel: 4x4
		# cell number: [200-600], [600-1000], [1000-1500]
		imageFileName = folder + 'imageSet_c8_k4_nb.dat'
		densityFileName = folder + 'densitySet_c8_k4_nb.dat'
		print(imageFileName)
		print(densityFileName)
		print('validation version:'+str(val_version))
		image_shape = (400,400)
		image_number = 300
		data = dl.train_data_load(imageFileName, image_shape, image_number);
		# normalize the data range into [0,255]
		for i in range(data.shape[0]):
			image = data[i,:,:]
			data[i,:,:] = 255*(image - np.min(image))/(np.max(image)-np.min(image))
		truthImages=dl.truth_data_load(densityFileName, image_shape, image_number);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data = data[0:200,:,:]
		test_data = data[200:300,:,:]
		train_truths = truthImages[0:200,:,:]
		test_truths = truthImages[200:300,:,:]

	if val_version==9:
		# 400 x 400 training data, cell size: 8 pixels, kernel: 4.5x4, intensity: x2
		# cell number: [200-600], [600-1000], [1000-1500]
		imageFileName = folder + 'imageSet_c8_k4_nb_v0.dat'
		densityFileName = folder + 'densitySet_c8_k4_nb_v0.dat'
		print(imageFileName)
		print(densityFileName)
		print('validation version:'+str(val_version))
		image_shape = (400,400)
		image_number = 300
		data = dl.train_data_load(imageFileName, image_shape, image_number);
		# normalize the data range into [0,255]
		for i in range(data.shape[0]):
			image = data[i,:,:]
			data[i,:,:] = 255*(image - np.min(image))/(np.max(image)-np.min(image))
		truthImages=dl.truth_data_load(densityFileName, image_shape, image_number);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data = data[0:200,:,:]
		test_data = data[200:300,:,:]
		train_truths = truthImages[0:200,:,:]
		test_truths = truthImages[200:300,:,:]

	if val_version==10:
		# 400 x 400 training data, cell size: 8 pixels, kernel: 5x4, intensity: x2
		# cell number: [200-600], [600-1000], [1000-1500]
		imageFileName = folder + 'imageSet_c8_k20_nb_v0.dat'
		densityFileName = folder + 'densitySet_c8_k20_nb_v0.dat'
		print(imageFileName)
		print(densityFileName)
		print('validation version:'+str(val_version))
		image_shape = (400,400)
		image_number = 300
		data = dl.train_data_load(imageFileName, image_shape, image_number);
		# normalize the data range into [0,255]
		for i in range(data.shape[0]):
			image = data[i,:,:]
			data[i,:,:] = 255*(image - np.min(image))/(np.max(image)-np.min(image))
		truthImages=dl.truth_data_load(densityFileName, image_shape, image_number);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data = data[0:200,:,:]
		test_data = data[200:300,:,:]
		train_truths = truthImages[0:200,:,:]
		test_truths = truthImages[200:300,:,:]

	if val_version==11:
		# 400 x 400 training data, cell size: 8 pixels, kernel: 5x4, intensity: x2
		# cell number: [200-600], [600-1000], [1000-1500]
		imageFileName = folder + 'imageSet_c8_k20_nb_v1.dat'
		densityFileName = folder + 'densitySet_c8_k20_nb_v1.dat'
		print(imageFileName)
		print(densityFileName)
		print('validation version:'+str(val_version))
		image_shape = (400,400)
		image_number = 300
		data = dl.train_data_load(imageFileName, image_shape, image_number);
		# normalize the data range into [0,255]
		for i in range(data.shape[0]):
			image = data[i,:,:]
			data[i,:,:] = 255*(image - np.min(image))/(np.max(image)-np.min(image))
		truthImages=dl.truth_data_load(densityFileName, image_shape, image_number);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data = data[0:200,:,:]
		test_data = data[200:300,:,:]
		train_truths = truthImages[0:200,:,:]
		test_truths = truthImages[200:300,:,:]

	if val_version==13:
                # 400 x 400 training data, cell size: 8 pixels, kernel: 5x4, intensity: x2
                # cell number: [200-600], [600-1000], [1000-1500]
                imageFileName = folder + 'imageSet_c8_k20_nb_v4.dat'
                densityFileName = folder + 'densitySet_c8_k20_nb_v4.dat'
                print(imageFileName)
                print(densityFileName)
                print('validation version:'+str(val_version))
                image_shape = (400,400)
                image_number = 300
                data = dl.train_data_load(imageFileName, image_shape, image_number);
                # normalize the data range into [0,255]
                for i in range(data.shape[0]):
                        image = data[i,:,:]
                        data[i,:,:] = 255*(image - np.min(image))/(np.max(image)-np.min(image))
                truthImages=dl.truth_data_load(densityFileName, image_shape, image_number);
#               data,truthImages=images_shuffle(data,truthImages)
                train_data = data[0:200,:,:]
                test_data = data[200:300,:,:]
                train_truths = truthImages[0:200,:,:]
                test_truths = truthImages[200:300,:,:]

	if val_version==12:
		# 400 x 400 training data, cell size: 8 pixels, kernel: 5x4, intensity: x2
		# cell number: [200-600], [600-1000]
		imageFileName = folder + 'imageSet_c8_k20_nb_v3.dat'
		densityFileName = folder + 'densitySet_c8_k20_nb_v3.dat'
		print(imageFileName)
		print(densityFileName)
		print('validation version:'+str(val_version))
		image_shape = (400,400)
		image_number = 200
		data = dl.train_data_load(imageFileName, image_shape, image_number);
		# normalize the data range into [0,255]
		for i in range(data.shape[0]):
			image = data[i,:,:]
			data[i,:,:] = 255*(image - np.min(image))/(np.max(image)-np.min(image))
		truthImages=dl.truth_data_load(densityFileName, image_shape, image_number);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data = data[0:100,:,:]
		test_data = data[100:200,:,:]
		train_truths = truthImages[0:100,:,:]
		test_truths = truthImages[100:200,:,:]

	if val_version==14:
		# 512x512 real images, kernel: 5x4
		imageFileName = folder + 'realImages_v1.dat'
		densityFileName = folder + 'realDensities_v1.dat'
		print(imageFileName)
		print(densityFileName)
		print('validation version:'+str(val_version))
		image_shape = (512,512)
		image_number = 10
		data = dl.train_data_load(imageFileName, image_shape, image_number);
		# normalize the data range into [0,255]
		for i in range(data.shape[0]):
			image = data[i,:,:]
			data[i,:,:] = 255*(image - np.min(image))/(np.max(image)-np.min(image))
		truthImages=dl.truth_data_load(densityFileName, image_shape, image_number);
# 		data,truthImages=images_shuffle(data,truthImages)
		# train data: 7 images validation data: 3 images
		train_data = data[0:7,:,:]
		test_data = data[7:10,:,:]
		train_truths = truthImages[0:7,:,:]
		test_truths = truthImages[7:10,:,:]

	if val_version==15:
		# 400 x 400 training data, cell size: 8 pixels, kernel: 5x4, intensity: x2
		# cell number: [200-600], [600-1000], [1000-1500]
		imageFileName = folder + 'imageSet_c9_k13.5_b_v4.dat'
		densityFileName = folder + 'densitySet_c9_k13.5_b_v4.dat'
		print(imageFileName)
		print(densityFileName)
		print('validation version:'+str(val_version))
		image_shape = (256,256)
		image_number = 200
		data = dl.train_data_load(imageFileName, image_shape, image_number);
		# normalize the data range into [0,255]
		for i in range(data.shape[0]):
			image = data[i,:,:]
			data[i,:,:] = 255*(image - np.min(image))/(np.max(image)-np.min(image))
		truthImages=dl.truth_data_load(densityFileName, image_shape, image_number);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data = data[0:100,:,:]
		test_data = data[100:200,:,:]
		train_truths = truthImages[0:100,:,:]
		test_truths = truthImages[100:200,:,:]

	if val_version==16:
		# 400 x 400 training data, cell size: 8 pixels, kernel: 5x4, intensity: x2
		# cell number: [200-600], [600-1000], [1000-1500]
		imageFileName = folder + 'imageSet_c9_k13.5_b_v4.dat'
		densityFileName = folder + 'densitySet_c9_k13.5_b_v4.dat'
		print(imageFileName)
		print(densityFileName)
		print('validation version:'+str(val_version))
		image_shape = (256,256)
		image_number = 200
		data = dl.train_data_load(imageFileName, image_shape, image_number);
		# normalize the data range into [0,255]
		for i in range(data.shape[0]):
			image = data[i,:,:]
			data[i,:,:] = 255*(image - np.min(image))/(np.max(image)-np.min(image))
		truthImages=dl.truth_data_load(densityFileName, image_shape, image_number);
# 		data,truthImages=images_shuffle(data,truthImages)
		train_data = data[0:100,:,:]
		test_data = data[100:130,:,:]
		train_truths = truthImages[0:100,:,:]
		test_truths = truthImages[100:130,:,:]

	return train_data, test_data, train_truths, test_truths

def images_shuffle(train_imgs,truth_imgs):
	shp=truth_imgs.shape
	image_set=np.concatenate((train_imgs.reshape((shp[0],shp[1],shp[2],1)),truth_imgs.reshape((shp[0],shp[1],shp[2],1))),axis=3)
	np.random.shuffle(image_set)
	train_imgs=image_set[:,:,:,0]
	truth_imgs=image_set[:,:,:,1]
	return train_imgs, truth_imgs

def save_all_results(model_path,loss_tr, loss_te, acc_tr, acc_te):
	generate_folder(model_path)
	f_loss_tr=model_path+'/loss_tr.txt'
	f_loss_te=model_path+'/loss_te.txt'
	f_val_tr=model_path+'/val_tr.txt'
	f_val_te=model_path+'/val_te.txt'
	np.savetxt(f_loss_tr,loss_tr);
	np.savetxt(f_loss_te,loss_te);
	np.savetxt(f_val_tr,acc_tr);
	np.savetxt(f_val_te,acc_te);

def save_train_loss(model_path,loss_tr, loss_te):
	generate_folder(model_path)
	f_loss_tr=model_path+'/loss_tr.txt'
	f_loss_te=model_path+'/loss_te.txt'
# 	f_val_tr=model_path+'/val_tr.txt'
# 	f_val_te=model_path+'/val_te.txt'
	np.savetxt(f_loss_tr,loss_tr);
	np.savetxt(f_loss_te,loss_te);
# 	np.savetxt(f_val_tr,acc_tr);
# 	np.savetxt(f_val_te,acc_te);
	
def save_test_result(model_path, nb_params, acc_tr):
	generate_folder(model_path)
	ls=[nb_params]+acc_tr
# 	ls=[nb_params, acc_tr]
	f_te=os.path.join(model_path, 'params_acc.txt')
# 	print(ls,f_te)
	np.savetxt(f_te,ls);

def save_model(model,model_name):
	generate_folder(model_name)
	model_path=model_name+'/model.h5'
	model.save(model_path)

# create the class weight for unbalanced training samples
def create_class_weights(y_train):
	class_weights=class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
	return class_weights

def gray_to_rgb(image):
	shp=image.shape
	rgb_images=np.zeros((shp[0],shp[1],3))
	for i in range(shp[0]):
		for j in range(shp[1]):
			if len(shp) ==2:
				if image[i,j]==1:
					rgb_images[i,j,:]=np.array([255,0,0])
				if image[i,j]==2:
					rgb_images[i,j,:]=np.array([0,100,0])
				if image[i,j]==3:
					rgb_images[i,j,:]=np.array([0,255,0])
				if image[i,j]==4:
					rgb_images[i,j,:]=np.array([255,255,255])
			else:
				if image[i,j,1]==1:
					rgb_images[i,j,:]=np.array([255,0,0])
				if image[i,j,1]==2:
					rgb_images[i,j,:]=np.array([0,100,0])
				if image[i,j,1]==3:
					rgb_images[i,j,:]=np.array([0,255,0])
				if image[i,j,1]==4:
					rgb_images[i,j,:]=np.array([255,255,255])
	return rgb_images

def zero_padding(image, pad_width):
	import numpy as np
	return np.pad(image,((pad_width[0],pad_width[1]),(pad_width[2],pad_width[3]),(0,0)),'constant')

## given the batch and hImageSize, nb_sampling, generate the patches
def patch_density_gen(batch, hImgSize, nb_sampling=100, test_flag=False):
	import numpy as np
	import random
	imglist = []
	densitylist = []
	
	shp = batch.shape
	for i in range(batch.shape[0]):	
		index_tuples=np.nonzero(batch[i,:,:,0])
		nb_pixels=index_tuples[0].shape[0]
		if test_flag==True:
			idx_set=range(nb_pixels)
		else:
			idx_set=random.sample(range(nb_pixels),nb_sampling)
		for cidx in idx_set:
			xid=index_tuples[0][cidx]
			yid=index_tuples[1][cidx]
			xlIdx=max(0,xid-hImgSize)
			xrIdx=min(shp[1]-1,xid+hImgSize)
			ylIdx=max(0,yid-hImgSize)
			yrIdx=min(shp[2]-1,yid+hImgSize)
			ax=xlIdx+hImgSize-xid
			bx=xid+hImgSize-xrIdx
			ay=ylIdx+hImgSize-yid
			by=yid+hImgSize-yrIdx
			extr_image=batch[i,xlIdx:xrIdx,ylIdx:yrIdx,:]
			image=np.pad(extr_image,((ax,bx),(ay,by),(0,0)),'constant')
			## normalize the each patch
# 			image = (image-np.mean(image))/np.std(image)
			im = image[:,:,0]
# 			im = (im-np.mean(im))/np.std(im)
			imglist.append(np.reshape(im,(image.shape[0],image.shape[1],1)))
			densitylist.append(np.reshape(image[:,:,1],(image.shape[0],image.shape[1],1)))
	return imglist, densitylist

def train_model(model, X_train, X_val, Y_train, Y_val, nb_epochs=400, nb_epoch_per_record=1, input_shape=(100,100,1), batch_size =256):
	from math import floor
	Y_train=Y_train.reshape((Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))
	X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
	Images=np.concatenate([X_train,Y_train, Y_train],axis=3)

	Y_val=Y_val.reshape((Y_val.shape[0],Y_val.shape[1],Y_val.shape[2],1))
	X_val=X_val.reshape((X_val.shape[0],X_val.shape[1],X_val.shape[2],1))
	val_Images=np.concatenate([X_val,Y_val, Y_val],axis=3)
	
	## train data generator
	train_datagen = ImageDataGenerator(
	horizontal_flip = True,
	vertical_flip =True,
	fill_mode = "constant",
	cval=0,
	zoom_range = 0.0,
	width_shift_range = 0,
	height_shift_range= 0,
	rotation_range=50)
	train_datagen.fit(Images)
	train_gen=train_datagen.flow(Images,None,batch_size=Images.shape[0])
	## validation data generator
	val_datagen = ImageDataGenerator(
	horizontal_flip = True,
	vertical_flip =True,
	fill_mode = "constant",
	cval=0,
	zoom_range = 0.0,
	width_shift_range = 0,
	height_shift_range=0,
	rotation_range=50)
	val_datagen.fit(val_Images)
	val_gen=val_datagen.flow(val_Images,None,batch_size=val_Images.shape[0])
	
	# training
	nb_sampling=200
	nb_val_sampling=200
	epochIdx=0
	tr_loss=[]
	te_loss=[]
	tr_acc =[]
	te_acc =[]
	mae_list=[]
	std_list=[]
	bs_acc =0.0
	bs_mse =float('Inf')
# 	print(input_shape[0])
	hImgSize=floor(input_shape[0]/2)
	print(hImgSize)
	while(epochIdx<nb_epochs):
		epochIdx+=1
		batch=train_gen.next()
		imglist, densitylist= patch_density_gen(batch[:,:,:,:2], hImgSize, nb_sampling=nb_sampling)
		trainImages=np.array(imglist)
		imglist=None
		densityMaps = np.array(densitylist)
		densitylist =None
		hist=model.fit(trainImages,densityMaps,batch_size=batch_size, nb_epoch=nb_epoch_per_record, verbose=1, shuffle=True)
		trainImages=None
		densityMaps = None
		## evaluation
		val_batch=val_gen.next()
		val_imglist, val_densitylist=patch_density_gen(val_batch[:,:,:,:2], hImgSize, nb_sampling=nb_val_sampling)
		valImages=np.array(val_imglist)
		valDensityMaps = np.array(val_densitylist)
		val_imglist = None
		val_densitylist = None
		score=model.evaluate(valImages,valDensityMaps,verbose=1,batch_size=int(batch_size))
		# make prediction
# 		mean_error =0
# 		std_error =0
# 		estimate_ls = []
# 		real_count_ls =[]
# 		batch_idx = 0
# 		valImages.shape
# 		while(True):
# 			if ((batch_idx+1)*batch_size<=valDensityMaps.shape[0]):
# 				image_arr = valImages[batch_idx*batch_size:(batch_idx+1)*batch_size,:]
# 				density_arr = valDensityMaps[batch_idx*batch_size:(batch_idx+1)*batch_size,:]
# 			elif ((batch_idx+1)*batch_size>valDensityMaps.shape[0] and batch_idx*batch_size<valDensityMaps.shape[0]):
# 				image_arr = valImages[batch_idx*batch_size:,:]
# 				density_arr = valDensityMaps[batch_idx*batch_size:,:]
# 			else:
# 				break
# 			preds = model.predict(image_arr)
# 			preds = preds.reshape(-1,preds.shape[1],preds.shape[2])
# 			preds = preds/100
# 			density_arr = density_arr/100
# 			pred_counts = np.apply_over_axes(np.sum,preds,[1,2]).reshape(preds.shape[0])
# 			ground_counts = np.apply_over_axes(np.sum,density_arr,[1,2]).reshape(preds.shape[0])
# 			estimate_ls +=pred_counts.tolist()
# 			real_count_ls +=ground_counts.tolist()	
# 			batch_idx += 1
# 		count_arr = np.array(estimate_ls)
# 		real_count_arr =np.array(real_count_ls)
# 		mean_error = np.mean(np.abs(count_arr-real_count_arr))
# 		std_error = np.std(np.abs(count_arr-real_count_arr))
# 		mae_list.append(mean_error)
# 		std_list.append(std_error)
		valImages=None
		valDensityMaps =None
		tr_loss.append(hist.history['loss'][-1])
		te_loss.append(score)
		print('\nepoch '+str(epochIdx)+'-> train mse:'+str(tr_loss[-1])+', val mse:'+str(te_loss[-1])+'--')
# 		print('epoch '+str(epochIdx)+'-> val mae:'+str(mae_list[-1])+', val std ae:'+str(std_list[-1])+'--')
		plot_loss(model.name, tr_loss, te_loss)
# 		plot_mae(model.name, mae_list, std_list)
		save_train_loss(model.name, tr_loss, te_loss)
# 		plot_save(model.name,tr_loss,te_loss,tr_acc,te_acc)
# 		save_all_results(model.name,tr_loss,te_loss,tr_acc,te_acc)
		if bs_mse>score:
			save_model(model,model.name)
			bs_mse=score

## train a linear regression model, without deconvolution
def train_regression_model(model, X_train, X_val, Y_train, Y_val, nb_epochs=400, nb_epoch_per_record=1, input_shape=(100,100,1), batch_size =256):
	from math import floor
	Y_train=Y_train.reshape((Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))
	X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
	Images=np.concatenate([X_train,Y_train, Y_train],axis=3)

	Y_val=Y_val.reshape((Y_val.shape[0],Y_val.shape[1],Y_val.shape[2],1))
	X_val=X_val.reshape((X_val.shape[0],X_val.shape[1],X_val.shape[2],1))
	val_Images=np.concatenate([X_val,Y_val, Y_val],axis=3)
	
	## train data generator
	train_datagen = ImageDataGenerator(
	horizontal_flip = True,
	vertical_flip =True,
	fill_mode = "constant",
	cval=0,
	zoom_range = 0.0,
	width_shift_range = 0,
	height_shift_range= 0,
	rotation_range=50)
	train_datagen.fit(Images)
	train_gen=train_datagen.flow(Images,None,batch_size=Images.shape[0])
	## validation data generator
	val_datagen = ImageDataGenerator(
	horizontal_flip = True,
	vertical_flip =True,
	fill_mode = "constant",
	cval=0,
	zoom_range = 0.0,
	width_shift_range = 0,
	height_shift_range=0,
	rotation_range=50)
	val_datagen.fit(val_Images)
	val_gen=val_datagen.flow(val_Images,None,batch_size=val_Images.shape[0])
	
	# training
	nb_sampling=200
	nb_val_sampling=200
	epochIdx=0
	tr_loss=[]
	te_loss=[]
	tr_acc =[]
	te_acc =[]
	bs_acc =0.0
	bs_mse =float('Inf')
# 	print(input_shape[0])
	hImgSize=floor(input_shape[0]/2)
	print(hImgSize)
	while(epochIdx<nb_epochs):
		epochIdx+=1
		batch=train_gen.next()
		imglist, densitylist= patch_density_gen(batch[:,:,:,:2], hImgSize, nb_sampling=nb_sampling)
		trainImages=np.array(imglist)
		imglist=None
		densityMaps = np.array(densitylist)
		densitylist =None
		# convert density map into the cell count
		cellVec = np.apply_over_axes(np.sum, densityMaps, [1,2]).reshape(-1,1)
		print(cellVec.shape)
		hist=model.fit(trainImages,cellVec,batch_size=batch_size, nb_epoch=nb_epoch_per_record, verbose=1, shuffle=True)
		trainImages=None
		densityMaps = None
		cellVec = None
		## evaluation
		val_batch=val_gen.next()
		val_imglist, val_densitylist=patch_density_gen(val_batch[:,:,:,:2], hImgSize, nb_sampling=nb_val_sampling)
		valImages=np.array(val_imglist)
		valDensityMaps = np.array(val_densitylist)
		val_imglist = None
		val_densitylist = None
		val_cellVec = np.apply_over_axes(np.sum, valDensityMaps, [1,2]).reshape(-1,1)
		score=model.evaluate(valImages,val_cellVec,verbose=1,batch_size=int(batch_size))
		valImages=None
		valDensityMaps =None
		val_cellVec = None
		tr_loss.append(hist.history['loss'][-1])
		te_loss.append(score)
# 		print([tr_loss[-1],te_loss[-1]])
		print('--train mse:'+str(tr_loss[-1])+', val mse:'+str(te_loss[-1])+'--')
		plot_loss(model.name, tr_loss, te_loss)
		save_train_loss(model.name, tr_loss, te_loss)
# 		plot_save(model.name,tr_loss,te_loss,tr_acc,te_acc)
# 		save_all_results(model.name,tr_loss,te_loss,tr_acc,te_acc)
		if bs_mse>score:
			save_model(model,model.name)
			bs_mse=score

# generate the test patches
def patch_gen_for_test(batch, imageSize, nb_sampling=9, test_flag=True):
	import numpy as np
	import random
	from math import floor
	imglist = []
	densitylist = []
	hImgSize=floor(imageSize/2)
	shp = batch.shape
	center = floor(shp[1]/2)
	for i in range(batch.shape[0]):
		# crop image (1,1)
		image = batch[i,:imageSize,:imageSize,:]
		imglist.append(np.reshape(image[:,:,0],(image.shape[0],image.shape[1],1)))
		densitylist.append(np.reshape(image[:,:,1],(image.shape[0],image.shape[1],1)))
		# 		print(image.shape)
		# crop image (1,2)
		image = batch[i,:imageSize,center-hImgSize:center+hImgSize,:]
		imglist.append(np.reshape(image[:,:,0],(image.shape[0],image.shape[1],1)))
		densitylist.append(np.reshape(image[:,:,1],(image.shape[0],image.shape[1],1)))
		# 		print(image.shape)
		# crop image (1,3)
		image = batch[i,:imageSize,shp[1]-imageSize:,:]
		imglist.append(np.reshape(image[:,:,0],(image.shape[0],image.shape[1],1)))
		densitylist.append(np.reshape(image[:,:,1],(image.shape[0],image.shape[1],1)))
		# 		print(image.shape)
		# crop image (2,1)
		image = batch[i,center-hImgSize:center+hImgSize,:imageSize,:]
		imglist.append(np.reshape(image[:,:,0],(image.shape[0],image.shape[1],1)))
		densitylist.append(np.reshape(image[:,:,1],(image.shape[0],image.shape[1],1)))
		# 		print(image.shape)		
		# crop image (2,2)
		image = batch[i,center-hImgSize:center+hImgSize,center-hImgSize:center+hImgSize,:]
		imglist.append(np.reshape(image[:,:,0],(image.shape[0],image.shape[1],1)))
		densitylist.append(np.reshape(image[:,:,1],(image.shape[0],image.shape[1],1)))
		# 		print(image.shape)			
		# crop image (2,3)
		image = batch[i,center-hImgSize:center+hImgSize,shp[1]-imageSize:,:]
		imglist.append(np.reshape(image[:,:,0],(image.shape[0],image.shape[1],1)))
		densitylist.append(np.reshape(image[:,:,1],(image.shape[0],image.shape[1],1)))
		# 		print(image.shape)
		# crop image (3,1)
		image = batch[i,shp[1]-imageSize:,:imageSize,:]	
		imglist.append(np.reshape(image[:,:,0],(image.shape[0],image.shape[1],1)))
		densitylist.append(np.reshape(image[:,:,1],(image.shape[0],image.shape[1],1)))
		# 		print(image.shape)	
		# crop image (3,2)
		image = batch[i,shp[1]-imageSize:,center-hImgSize:center+hImgSize,:]
		imglist.append(np.reshape(image[:,:,0],(image.shape[0],image.shape[1],1)))
		densitylist.append(np.reshape(image[:,:,1],(image.shape[0],image.shape[1],1)))
		# 		print(image.shape)		
		# crop image (3,3)
		image = batch[i,shp[1]-imageSize:,shp[1]-imageSize:,:]
		imglist.append(np.reshape(image[:,:,0],(image.shape[0],image.shape[1],1)))
		densitylist.append(np.reshape(image[:,:,1],(image.shape[0],image.shape[1],1)))
	return imglist, densitylist

def patch_merge_for_display(imgSet, imgSize):
	from math import floor
	shp = imgSet.shape
	imgNum = shp[0]
	imgLen = shp[1]
	# zero padding the estimated density map patch
	images = np.zeros((imgNum,imgSize,imgSize))
	hImgSize = floor(imgLen/2)
	center = floor(imgSize/2)
	images[0,:imgLen,:imgLen] = imgSet[0]
	images[1,:imgLen,center-hImgSize:center+hImgSize] = imgSet[1]
	images[2,:imgLen,imgSize-imgLen:] = imgSet[2]
	images[3,center-hImgSize:center+hImgSize,:imgLen] = imgSet[3]
	images[4,center-hImgSize:center+hImgSize,center-hImgSize:center+hImgSize] = imgSet[4]
	images[5,center-hImgSize:center+hImgSize,imgSize-imgLen:] = imgSet[5]
	images[6,imgSize-imgLen:,:imgLen] = imgSet[6]
	images[7,imgSize-imgLen:,center-hImgSize:center+hImgSize] = imgSet[7]
	images[8,imgSize-imgLen:,imgSize-imgLen:] = imgSet[8]
	# merge the estimated density map patch
	merged_image = np.zeros((imgSize, imgSize))
	weight_map = np.zeros((imgSize, imgSize))
	weight_map = np.sum((images>0)*1,axis = 0)
# 	index_tuples=np.nonzero(weight_map)
# 	nb_pixels=index_tuples[0].shape[0]
	index_tuples=np.where(weight_map==0)
	nb_pixels=index_tuples[0].shape[0]
	for i in range(nb_pixels):
		xid = index_tuples[0][i]
		yid = index_tuples[1][i]
		weight_map[xid,yid] = 1
	map = 1.0/weight_map
	for i in range(imgNum):
		merged_image += images[i,:,:]*map
	return merged_image

# zero pad with an same extension width to each direction
def image_zeropad(image, exWidth):
	return np.lib.pad(image,((exWidth,exWidth),(exWidth,exWidth)),'constant')

# shrink an image with the same shrinkage length to each direction
def image_shrink(image, shLength):
	shp = image.shape
	return image[shLength:shp[0]-shLength,shLength:shp[1]-shLength]
	
# zero pad an image set
def image_arr_zeropad(imgSet, exWidth):
	shp = imgSet.shape
	if len(shp)<3:
		print('shrink input error!!!')
		return
	return np.lib.pad(imgSet,((0,0),(exWidth,exWidth),(exWidth,exWidth)),'constant')

# shrink an image set
def image_arr_shrink(imgSet, shLength):
	shp = imgSet.shape
	if len(shp)<3:
		print('shrink input error!!!')
		return
	return imgSet[:,shLength:shp[1]-shLength,shLength:shp[2]-shLength]

# depatch an image set
def image_depatch(imgSet, output_size):
	from math import floor
	shp = imgSet.shape
	imgNum = shp[0]
	imgSize = shp[1]
	if imgSize%output_size != 0:
		pad_size = int(imgSize/output_size+1)*output_size
		zero_thickness = int((pad_size- imgSize)/2)
		imgSet = image_arr_zeropad(imgSet, zero_thickness)

	imgSize = imgSet.shape[1]
	nb_patch_x = int(imgSize/output_size)
	patch_list=[]
	for i in range(imgNum):
		image = imgSet[i,:,:]
		for px in range(nb_patch_x):
			for py in range(nb_patch_x):
				patch_list.append(image[px*output_size:(px+1)*output_size,py*output_size:(py+1)*output_size])
	# 				
	# 	patch_arr = np.array(patch_list,(-1,output_size,output_size))
	return np.array(patch_list)
	
# merge an image set
def image_merge(preds, output_size):
	shp = preds.shape
	merged_size = output_size
	if output_size%shp[1]!= 0:
		merged_size = (int(output_size/shp[1])+1)*shp[1]
	nb_x = int(merged_size/shp[1])
	nb_pch_per_img = nb_x*nb_x
	den_list = []
	nb_map = int(shp[0]/nb_pch_per_img)
	den_map = np.zeros((nb_map,merged_size,merged_size))
	for i in range(shp[0]):
		mIdx = i//nb_pch_per_img
		kth = i%nb_pch_per_img
		xth = kth//nb_x
		yth = kth%nb_x
		den_map[mIdx,xth*shp[1]:(xth+1)*shp[1],yth*shp[1]:(yth+1)*shp[1]]=preds[i,:,:]
	# shrink the images
	shLength = int((merged_size-output_size)/2)
	den_map = image_arr_shrink(den_map, shLength)
	return den_map

def img_den_pair_depatch(batch, output_size):
	from math import floor
	imgSet = batch
	shp = imgSet.shape
	imgNum = shp[0]
	imgSize = shp[1]
	if imgSize%output_size != 0:
		pad_size = int(imgSize/output_size+1)*output_size
		zero_thickness = int((pad_size- imgSize)/2)
		imgSet = image_arr_zeropad(imgSet, zero_thickness)

	imgSize = imgSet.shape[1]
	nb_patch_x = int(imgSize/output_size)
	patch_list=[]
	for i in range(imgNum):
		image = imgSet[i,:,:]
		for px in range(nb_patch_x):
			for py in range(nb_patch_x):
				patch_list.append(image[px*output_size:(px+1)*output_size,py*output_size:(py+1)*output_size])
	patch_arr = np.array(patch_list)
	image_patches = patch_arr[:,:,:,0]
	density_patches = patch_arr[:,:,:,1]

	return image_patches, density_patches


def test_model(model,X_data,Y_data, nb_image, input_shape=(65,65,3)):
	pixelList=[]
	pixelLabelList=[]
	hImgSize=floor(input_shape[0]/2)
	
	# training data
	for i in range(nb_image):
	# for i in range(1):
		image=X_data[i,:,:,:]
		ground_truth=Y_data[i,:,:]
		image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
		ground_truth=ground_truth.reshape(1,ground_truth.shape[0],ground_truth.shape[1],1)
		
# 		Y_val=Y_val.reshape((Y_data.shape[0],Y_data.shape[1],Y_data.shape[2],1))
		val_Image=np.concatenate([image,ground_truth],axis=3)
# 		patchList=[]
# 		truthList=[]
# 		for j in range(60,350):
# 			for k in range(60,350):
# 					if not np.amax(image[j,k,:])==0:
# 						patchList.append(image[j-hImgSize:j+hImgSize+1,k-hImgSize:k+hImgSize+1,:])
# 						truthList.append(Y_data[i,j,k])
# 		print('## testing data loding ##')
		patchList, truthList=patch_label_gen(val_Image, hImgSize, test_flag=True)
# 		print('## testing data loaded! ##')
		patch_arr=np.array(patchList)
		truth_arr=np.array(truthList)
		truthLabels=label_binarize(truthList,classes=[0,1,2,3,4])
		preds=model.predict(patch_arr,verbose=0)
		score=model.evaluate(patch_arr,truthLabels,verbose=1,batch_size=100)
		pred_list=[]
		for pred in preds:
			idx_max=np.argmax(pred)
			pred_list.append(idx_max)
	# 	target_names = ['class 0','class 1', 'class 2', 'class 3', 'class 4']
	# 	print(classification_report(truth_arr, pred_list, target_names=target_names))
		pixelList+=pred_list
		pixelLabelList+=truthList
	cmat=confusion_matrix(pixelLabelList,pixelList)
	acc_arr=cmat.diagonal()/cmat.sum(axis=1)
	acc_ave=cmat.diagonal().sum()/cmat.sum()
	acc_list=acc_arr.tolist()
	acc_list.append(acc_ave)  # accuracy
	ratios=[pixelLabelList.count(lb)/len(pixelLabelList) for lb in range(5)]
	print(acc_ave)
	return acc_list, ratios
	
## The tool for cnn_generator
def get_crop_shape(target, refer):
	# width, the 3rd dimension
	cw = (target.get_shape()[2]-refer.get_shape()[2]).value
	assert(cw>0)
	if cw%2 !=0:
		cw1, cw2 = int(cw/2), int(cw/2)+1
	else:
		cw1, cw2 = int(cw/2), int(cw/2)
	# height, the 2nd dimension
	ch = (target.get_shape()[1]-refer.get_shape()[1]).value	
# 	assert(ch>0)
# 	if ch%2 !=0:
# 		ch1, ch2 = int(ch/2), int(ch/2)+1
# 	else:
# 		ch1, ch2 = int(ch/2), int(ch/2)
	ch1, ch2 = cw1, cw2
	return (ch1, ch2), (cw1, cw2)

## display the results as a video
## input: figure instance, image set, prediction set and ground truth density map
def visualize_results(fig, imgSet, predSet, grdSet, nb_slides, interval):
	import numpy as np
	import matplotlib.pyplot as plt
	for i in range(nb_slides):
		plt.clf()
		# read the image
		shp = imgSet.shape
		image = imgSet[i,:].reshape(shp[1],shp[2])
		ground_truth = grdSet[i,:].reshape(shp[1],shp[2])
		max_value = np.max(ground_truth)
		min_value = np.min(ground_truth)
		pred_map = predSet[i,:].reshape(shp[1],shp[2])
		pred_count = np.sum(pred_map)
		real_count = np.sum(ground_truth)
		pred_map = (pred_map - np.min(pred_map))/(np.max(pred_map)-np.min(pred_map))*(max_value-min_value)+min_value
		print([pred_count, real_count])
		ax = fig.add_subplot(1,3,1)
		cax = ax.imshow(image)
		fig.colorbar(cax)
		ax.set_title('Cell image('+str(shp[1])+'x'+str(shp[2])+')')
		ax = fig.add_subplot(1,3,2)
		cax = ax.imshow(pred_map)
		fig.colorbar(cax)
		ax.set_title('Estimated density('+str(shp[1])+'x'+str(shp[2])+')')
		ax.set_xlabel('Cell count:'+str(pred_count))
		ax = fig.add_subplot(1,3,3)
		cax = ax.imshow(ground_truth)
		fig.colorbar(cax)
		ax.set_title('Ground truth('+str(shp[1])+'x'+str(shp[2])+')')
		ax.set_xlabel('Cell count:'+str(real_count))
		plt.pause(interval)