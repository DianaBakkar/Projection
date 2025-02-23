##################First attempt to make projections of image stack into a coordinate system############
###############Reconstruct the image stack from data.bin file###########
import numpy as np
#Load ops.npy to get metadata#
ops=np.load('ops.npy',allow_pickle=True).item()
Ly,Lx=ops['Ly'],ops['Lx'] #Image dimensions
n_frames=ops['nframes']   #Number of frames

#Load the binary file#
data=np.memmap('data.bin',dtype='int16',mode='r',shape=(n_frames,Ly,Lx))

#Generate projections#
mean_projection=np.mean(data,axis=0) #Averages pixel intensity over all frames (smooth representation)
max_projection=np.max(data,axis=0)   #Displays highest intensity per pixel (enhances bright structures)

#Display the mean projections#
import matplotlib.pyplot as plt
plt.imshow(mean_projection,cmap='gray') #Display the mean projection as a graysacle image
plt.title("Mean projection from data.bin")
plt.colorbar()
plt.show()





