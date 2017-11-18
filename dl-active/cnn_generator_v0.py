# cnn generator for 'grid' searching for some task
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Input, Cropping2D, concatenate
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,GlobalAveragePooling2D, Conv2D,Lambda, UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU, ELU

def fcn32():
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	nb_dense=512
	dropout=0.5
	lr=0.0001

	inputs = Input(shape=(100,100,1))
	x = Conv2D(nb_filters, kernel_size, padding='same')(inputs)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=pool_size)(x)

	x = Conv2D(nb_filters, kernel_size, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=pool_size)(x)

	x = Conv2D(nb_filters*2, kernel_size, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=pool_size)(x)

	x = Conv2D(nb_filters*4, kernel_size, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(nb_filters*16, kernel_size, padding='same')(x)
	x = BatchNormalization()(x)
	y = Activation('relu')(x)



	x = Conv2DTranspose(nb_filters*4, kernel_size, strides=(1, 1), padding='same')(y)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = UpSampling2D(size=pool_size)(x)
	#     x = ZeroPadding2D(padding=(1, 1))(x)

	x = Conv2DTranspose(nb_filters*2, kernel_size, strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)    
	x = UpSampling2D(size=pool_size)(x)
	x = ZeroPadding2D(padding=(1, 1))(x)

	x = Conv2DTranspose(nb_filters, kernel_size, strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)    
	x = UpSampling2D(size=pool_size)(x)

	x = Conv2D(1, kernel_size, padding='same')(x)

	model = Model(inputs, x)
    
	return model	

def fcn32A():
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	nb_dense=512
	dropout=0.5
	lr=0.0001
	
	inputs = Input(shape=(100,100,1))
	x = Conv2D(nb_filters, kernel_size, padding='same')(inputs)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=pool_size)(x)

	x = Conv2D(nb_filters, kernel_size, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=pool_size)(x)

	x = Conv2D(nb_filters*2, kernel_size, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=pool_size)(x)

	x = Conv2D(nb_filters*4, kernel_size, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(nb_filters*16, kernel_size, padding='same')(x)
	x = BatchNormalization()(x)
	y = Activation('relu')(x)


	x = UpSampling2D(size=pool_size)(y)
	x = Conv2DTranspose(nb_filters*4, kernel_size, strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	#     x = ZeroPadding2D(padding=(1, 1))(x)

	x = UpSampling2D(size=pool_size)(x)
	x = ZeroPadding2D(padding=(1, 1))(x)
	x = Conv2DTranspose(nb_filters*2, kernel_size, strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)    

	x = UpSampling2D(size=pool_size)(x)
	x = Conv2DTranspose(nb_filters, kernel_size, strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)    

	x = Conv2D(1, kernel_size, padding='same')(x)

	model = Model(inputs, x)
    
	return model
	
def fcn200():
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	nb_dense=512
	dropout=0.5
	lr=0.0001

	inputs = Input(shape=(200,200,1))
	x = Conv2D(nb_filters, kernel_size, padding='same')(inputs)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=pool_size)(x)

	x = Conv2D(nb_filters, kernel_size, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=pool_size)(x)

	x = Conv2D(nb_filters*2, kernel_size, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=pool_size)(x)

	x = Conv2D(nb_filters*2, kernel_size, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(pool_size=pool_size)(x)

	x = Conv2D(nb_filters*4, kernel_size, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(nb_filters*16, kernel_size, padding='same')(x)
	x = BatchNormalization()(x)
	y = Activation('relu')(x)


	x = UpSampling2D(size=pool_size)(y)
	x = Conv2DTranspose(nb_filters*4, kernel_size, strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	#     x = ZeroPadding2D(padding=(1, 1))(x)

	x = UpSampling2D(size=pool_size)(x)
	x = ZeroPadding2D(padding=(1, 1))(x)
	x = Conv2DTranspose(nb_filters*2, kernel_size, strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)    

	x = UpSampling2D(size=pool_size)(x)
# 	x = ZeroPadding2D(padding=(1, 1))(x)
	x = Conv2DTranspose(nb_filters*2, kernel_size, strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)    

	x = UpSampling2D(size=pool_size)(x)
	x = Conv2DTranspose(nb_filters, kernel_size, strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)    

	x = Conv2D(1, kernel_size, padding='same')(x)

	model = Model(inputs, x)
    
	return model

def FCN():
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	nb_dense=512
	dropout=0.5
	concat_axis = 3
	lr=0.0001

	import helper_functions as hf
	def get_crop_shape(target, refer):
		return hf.get_crop_shape(target, refer)

	inputs = Input(shape=(200,200,1))
	conv1 = Conv2D(nb_filters, kernel_size, padding='same')(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

	conv2 = Conv2D(nb_filters, kernel_size, padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

	conv3 = Conv2D(nb_filters*2, kernel_size, padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

	conv4 = Conv2D(nb_filters*2, kernel_size, padding='same')(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)
	pool4 = MaxPooling2D(pool_size=pool_size)(conv4)

	conv5 = Conv2D(nb_filters*4, kernel_size, padding='same')(pool4)
	conv5 = BatchNormalization()(conv5)
	conv5 = Activation('relu')(conv5)

	conv6 = Conv2D(nb_filters*16, kernel_size, padding='same')(conv5)
	conv6 = BatchNormalization()(conv6)
	conv6 = Activation('relu')(conv6)


	up_conv6 = UpSampling2D(size=pool_size)(conv6)
	ch, cw = get_crop_shape(conv4, up_conv6)
	crop_conv4 = Cropping2D(cropping = (ch,cw))(conv4)
	up7 = concatenate([up_conv6, crop_conv4], axis = concat_axis)
	de_conv7 = Conv2DTranspose(nb_filters*4, kernel_size, strides=(1, 1), padding='same')(up7)
	de_conv7 = BatchNormalization()(de_conv7)
	de_conv7 = Activation('relu')(de_conv7)
	#     x = ZeroPadding2D(padding=(1, 1))(x)

	up_conv7 = UpSampling2D(size=pool_size)(de_conv7)
	up_conv7 = ZeroPadding2D(padding=(1, 1))(up_conv7)
# 	ch, cw = get_crop_shape(conv3, up_conv8)
# 	crop_conv4 = Cropping2D(cropping = (ch,cw))(conv4)	
	up8 = concatenate([up_conv7, conv3], axis = concat_axis)
	de_conv8 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(1, 1), padding='same')(up8)
	de_conv8 = BatchNormalization()(de_conv8)
	de_conv8 = Activation('relu')(de_conv8)    

	up_conv8 = UpSampling2D(size=pool_size)(de_conv8)
# 	x = ZeroPadding2D(padding=(1, 1))(x)
	up9 = concatenate([up_conv8, conv2], axis = concat_axis)
	de_conv9 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(1, 1), padding='same')(up9)
	de_conv9 = BatchNormalization()(de_conv9)
	de_conv9 = Activation('relu')(de_conv9)    

	up_conv9 = UpSampling2D(size=pool_size)(de_conv9)
	up10 = concatenate([up_conv9, conv1], axis = concat_axis)
	de_conv10 = Conv2DTranspose(nb_filters, kernel_size, strides=(1, 1), padding='same')(up10)
	de_conv10 = BatchNormalization()(de_conv10)
	de_conv10 = Activation('relu')(de_conv10)    

	fcn_output = Conv2D(1, kernel_size, padding='same')(de_conv10)

	model = Model(inputs, fcn_output)
    
	return model
	
def FCN1():
	## add convolutional layers after concatenation of the features from different layers
	## modified at Oct-06-2017
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	nb_dense=512
	dropout=0.5
	concat_axis = 3
	lr=0.0001

	import helper_functions as hf
	def get_crop_shape(target, refer):
		return hf.get_crop_shape(target, refer)

	inputs = Input(shape=(200,200,1))
	conv1 = Conv2D(nb_filters, kernel_size, padding='same')(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

	conv2 = Conv2D(nb_filters, kernel_size, padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

	conv3 = Conv2D(nb_filters*2, kernel_size, padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

	conv4 = Conv2D(nb_filters*2, kernel_size, padding='same')(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)
	pool4 = MaxPooling2D(pool_size=pool_size)(conv4)

	conv5 = Conv2D(nb_filters*4, kernel_size, padding='same')(pool4)
	conv5 = BatchNormalization()(conv5)
	conv5 = Activation('relu')(conv5)

	conv6 = Conv2D(nb_filters*16, kernel_size, padding='same')(conv5)
	conv6 = BatchNormalization()(conv6)
	conv6 = Activation('relu')(conv6)


	up_conv6 = UpSampling2D(size=pool_size)(conv6)
	ch, cw = get_crop_shape(conv4, up_conv6)
	crop_conv4 = Cropping2D(cropping = (ch,cw))(conv4)
	up7 = concatenate([up_conv6, crop_conv4], axis = concat_axis)
	conv7 = Conv2D(nb_filters*16, kernel_size, padding='same')(up7)
	de_conv7 = Conv2DTranspose(nb_filters*4, kernel_size, strides=(1, 1), padding='same')(conv7)
	de_conv7 = BatchNormalization()(de_conv7)
	de_conv7 = Activation('relu')(de_conv7)
	#     x = ZeroPadding2D(padding=(1, 1))(x)

	up_conv7 = UpSampling2D(size=pool_size)(de_conv7)
	up_conv7 = ZeroPadding2D(padding=(1, 1))(up_conv7)
# 	ch, cw = get_crop_shape(conv3, up_conv8)
# 	crop_conv4 = Cropping2D(cropping = (ch,cw))(conv4)	
	up8 = concatenate([up_conv7, conv3], axis = concat_axis)
	conv8 = Conv2D(nb_filters*4, kernel_size, padding='same')(up8)
	de_conv8 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(1, 1), padding='same')(conv8)
	de_conv8 = BatchNormalization()(de_conv8)
	de_conv8 = Activation('relu')(de_conv8)    

	up_conv8 = UpSampling2D(size=pool_size)(de_conv8)
# 	x = ZeroPadding2D(padding=(1, 1))(x)
	up9 = concatenate([up_conv8, conv2], axis = concat_axis)
	conv9 = Conv2D(nb_filters*2, kernel_size, padding='same')(up9)
	de_conv9 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(1, 1), padding='same')(conv9)
	de_conv9 = BatchNormalization()(de_conv9)
	de_conv9 = Activation('relu')(de_conv9)    

	up_conv9 = UpSampling2D(size=pool_size)(de_conv9)
	up10 = concatenate([up_conv9, conv1], axis = concat_axis)
	conv10 = Conv2D(nb_filters, kernel_size, padding='same')(up10)
	de_conv10 = Conv2DTranspose(nb_filters, kernel_size, strides=(1, 1), padding='same')(conv10)
	de_conv10 = BatchNormalization()(de_conv10)
	de_conv10 = Activation('relu')(de_conv10)    

	fcn_output = Conv2D(1, kernel_size, padding='same')(de_conv10)

	model = Model(inputs, fcn_output)
    
	return model
	
def ResFCN():
	## add convolutional layers after concatenation of the features from different layers
	## add residual network
	## modified at Oct-31-2017
	nb_filters=32
	kernel_size=(3,3)
	pool_size=(2,2)
	nb_dense=512
	dropout=0.5
	concat_axis = 3
	lr=0.0001

	import helper_functions as hf
	def get_crop_shape(target, refer):
		return hf.get_crop_shape(target, refer)

	inputs = Input(shape=(200,200,1))
	conv1 = Conv2D(nb_filters, kernel_size, padding='same')(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	x = Conv2D(nb_filters, kernel_size, padding='same')(conv1)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	conv1 = merge([x,conv1],mode='sum')
	pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

	conv2 = Conv2D(nb_filters, kernel_size, padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	x = Conv2D(nb_filters, kernel_size, padding='same')(conv2)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	conv2 = merge([x,conv2],mode='sum')
	pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

	conv3 = Conv2D(nb_filters*2, kernel_size, padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	x = Conv2D(nb_filters*2, kernel_size, padding='same')(conv3)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	conv3 = merge([x,conv3],mode='sum')
	pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

	conv4 = Conv2D(nb_filters*2, kernel_size, padding='same')(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)
	x = Conv2D(nb_filters*2, kernel_size, padding='same')(conv4)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	conv4 = merge([x,conv4],mode='sum')
	pool4 = MaxPooling2D(pool_size=pool_size)(conv4)

	conv5 = Conv2D(nb_filters*4, kernel_size, padding='same')(pool4)
	conv5 = BatchNormalization()(conv5)
	conv5 = Activation('relu')(conv5)
	x = Conv2D(nb_filters*4, kernel_size, padding='same')(conv5)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	conv5 = merge([x,conv5],mode='sum')

	conv6 = Conv2D(nb_filters*16, kernel_size, padding='same')(conv5)
	conv6 = BatchNormalization()(conv6)
	conv6 = Activation('relu')(conv6)
	x = Conv2D(nb_filters*16, kernel_size, padding='same')(conv6)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	conv6 = merge([x,conv6],mode='sum')

	up_conv6 = UpSampling2D(size=pool_size)(conv6)
	ch, cw = get_crop_shape(conv4, up_conv6)
	crop_conv4 = Cropping2D(cropping = (ch,cw))(conv4)
	up7 = concatenate([up_conv6, crop_conv4], axis = concat_axis)
	conv7 = Conv2D(nb_filters*16, kernel_size, padding='same')(up7)
	de_conv7 = Conv2DTranspose(nb_filters*4, kernel_size, strides=(1, 1), padding='same')(conv7)
	de_conv7 = BatchNormalization()(de_conv7)
	de_conv7 = Activation('relu')(de_conv7)
	#     x = ZeroPadding2D(padding=(1, 1))(x)

	up_conv7 = UpSampling2D(size=pool_size)(de_conv7)
	up_conv7 = ZeroPadding2D(padding=(1, 1))(up_conv7)
# 	ch, cw = get_crop_shape(conv3, up_conv8)
# 	crop_conv4 = Cropping2D(cropping = (ch,cw))(conv4)	
	up8 = concatenate([up_conv7, conv3], axis = concat_axis)
	conv8 = Conv2D(nb_filters*4, kernel_size, padding='same')(up8)
	de_conv8 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(1, 1), padding='same')(conv8)
	de_conv8 = BatchNormalization()(de_conv8)
	de_conv8 = Activation('relu')(de_conv8)    

	up_conv8 = UpSampling2D(size=pool_size)(de_conv8)
# 	x = ZeroPadding2D(padding=(1, 1))(x)
	up9 = concatenate([up_conv8, conv2], axis = concat_axis)
	conv9 = Conv2D(nb_filters*2, kernel_size, padding='same')(up9)
	de_conv9 = Conv2DTranspose(nb_filters*2, kernel_size, strides=(1, 1), padding='same')(conv9)
	de_conv9 = BatchNormalization()(de_conv9)
	de_conv9 = Activation('relu')(de_conv9)    

	up_conv9 = UpSampling2D(size=pool_size)(de_conv9)
	up10 = concatenate([up_conv9, conv1], axis = concat_axis)
	conv10 = Conv2D(nb_filters, kernel_size, padding='same')(up10)
	de_conv10 = Conv2DTranspose(nb_filters, kernel_size, strides=(1, 1), padding='same')(conv10)
	de_conv10 = BatchNormalization()(de_conv10)
	de_conv10 = Activation('relu')(de_conv10)    

	fcn_output = Conv2D(1, kernel_size, padding='same')(de_conv10)

	model = Model(inputs, fcn_output)
    
	return model