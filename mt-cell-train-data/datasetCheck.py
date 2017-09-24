import matplotlib.pyplot as plt
import numpy as np

folder = './dataset/'

imageFileName = folder+'imageSet.dat'
densityFileName = folder+'densitySet.dat'
imageNum = 300
xPixels = 250
yPixels = 250

imageData = np.fromfile(imageFileName, dtype=float, count=xPixels*yPixels*imageNum, sep ='')
densityData = np.fromfile(densityFileName, dtype=float, count=xPixels*yPixels*imageNum, sep ='')

# reshape
imageSet = np.reshape(imageData,(imageNum,xPixels,yPixels))
densitySet = np.reshape(densityData,(imageNum,xPixels,yPixels))

plt.ion()
fig = plt.figure()
#plt.ion()


for i in range(imageNum):
    ax = fig.add_subplot(1,2,1)
    ax.set_title('Synthetic cells')
    image = imageSet[i,:,:]
    ax.imshow(image)
    ax = fig.add_subplot(1,2,2)
    ax.set_title('Density map')
    density = densitySet[i,:,:]
    ax.imshow(density)
    plt.pause(0.5)


