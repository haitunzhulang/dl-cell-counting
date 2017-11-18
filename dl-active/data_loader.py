import numpy as np
import random

train_size=300
tr_imgSize=250*250
truth_imgSize=250*250
pixelNums = 250

def train_load(train_path):
	TRAIN_SIZE=train_size
	PIXEL_NUM=tr_imgSize
	train_data=np.fromfile(train_path,dtype=np.float,count=TRAIN_SIZE*PIXEL_NUM,sep='')
	train_data=train_data.reshape((-1,pixelNums,pixelNums));

	return train_data

def truth_load(train_path):
	TRAIN_SIZE=train_size
	PIXEL_NUM=truth_imgSize
	train_data=np.fromfile(train_path,dtype=np.float,count=TRAIN_SIZE*PIXEL_NUM,sep='')
	train_data=train_data.reshape((-1,pixelNums,pixelNums));

	return train_data
	
def train_data_load(train_path, data_shape = (400,400), image_num = 300):
	TRAIN_SIZE=data_shape[0]*data_shape[1]
	PIXEL_NUM=image_num
	train_data=np.fromfile(train_path,dtype=np.float,count=TRAIN_SIZE*PIXEL_NUM,sep='')
	train_data=train_data.reshape((-1,data_shape[0],data_shape[1]));

	return train_data

def truth_data_load(train_path, data_shape = (400,400), image_num = 300):
	TRAIN_SIZE=data_shape[0]*data_shape[1]
	PIXEL_NUM=image_num
	train_data=np.fromfile(train_path,dtype=np.float,count=TRAIN_SIZE*PIXEL_NUM,sep='')
	train_data=train_data.reshape((-1,data_shape[0],data_shape[1]));

	return train_data