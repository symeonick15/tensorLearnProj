import numpy as np

featureDatasetDir = '/home/nick/Downloads/low_freq/house_1'
featureDatasetName = '/channel_1.dat'

labelDatasetDir = '/home/nick/Downloads/low_freq/house_1'
labelDatasetName = '/channel_5.dat'

arr = np.load('/home/nick/Downloads/low_freq/house_1/proc/channel_1.dat.npy')

print(arr)
print(len(arr))

arr = np.load('/home/nick/Downloads/low_freq/house_1/proc/channel_5.dat.npy')

print(arr)
print(len(arr))