import matplotlib.pyplot as plt
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb, [0.299, 0.587, 0.114])

imageFileName = 'image.dat'
densityFileName = 'density.dat'

image = np.fromfile(imageFileName, dtype=float, count=250*250*3, sep ='')
density = np.fromfile(densityFileName, dtype=float, count=250*250, sep ='')

# reshape
image = np.reshape(image,(3,250,250))
#image = image.transpose()
image = np.transpose(image,(1,2,0))
#if not np.amax(image) == 0:
#    image[:,:,0] = (image[:,:,0] - np.amin(image[:,:,0]))/(np.amax(image[:,:,0])-np.amin(image[:,:,0]))
#    image[:,:,1] = (image[:,:,1] - np.amin(image[:,:,1]))/(np.amax(image[:,:,1])-np.amin(image[:,:,1]))
#    image[:,:,2] = (image[:,:,2] - np.amin(image[:,:,2]))/(np.amax(image[:,:,2])-np.amin(image[:,:,2]))
#iimage = np.int8(image*255)

gray_img = rgb2gray(image)
density = np.reshape(density,(250,250))

fig = plt.figure()

ax = fig.add_subplot(1,2,1)
ax.set_title('Synthetic cells')
#ax.imshow(image[:,:,1])
ax.imshow(gray_img)
ax = fig.add_subplot(1,2,2)
ax.set_title('Density map')
ax.imshow(density)
plt.show()

