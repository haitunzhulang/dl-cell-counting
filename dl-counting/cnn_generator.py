# cnn generator for 'grid' searching for some task
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Input
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