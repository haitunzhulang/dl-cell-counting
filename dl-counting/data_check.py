import data_loader as dl
import matplotlib.pyplot as plt


folder = '/home/shenghua/dl-cell-counting/mt-cell-train-data/dataset/'

#imageFileName = folder + 'imageSet.dat'
#densityFileName = folder + 'densitySet.dat'

imageFileName = folder + 'realImages.dat'
densityFileName = folder + 'realDensities.dat'

# x = dl.train_load(imageFileName)
# y = dl.truth_load(densityFileName)

x = dl.train_data_load(imageFileName,(512,512),10)
y = dl.truth_data_load(densityFileName,(512,512),10)

fig = plt.figure()

ax = fig.add_subplot(1,2,1)
ax.imshow(x[0,:,:])
ax.set_title('Synthetic images')
ax = fig.add_subplot(1,2,2)
ax.imshow(y[0,:,:])
ax.set_title('Density map')
plt.show()


plt.ion()
fig = plt.figure()
for i in range(len(x)):
	ax = fig.add_subplot(1,2,1)
	ax.imshow(x[i])
	ax = fig.add_subplot(1,2,2)
	ax.imshow(y[i])
	plt.pause(0.5)
	
import keras.backend as K
lr=0.0005
model=cg.layer9_cnn()
sgd = SGD(lr=lr, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
# model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
