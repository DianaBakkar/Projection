##################First attempt to make projections of image stack into a coordinate system############
###############Reconstruct the image stack from data.bin file###########
import numpy as np
#Load ops.npy to get metadata#
ops=np.load('ops.npy',allow_pickle=True).item()
stat=np.load('stat_plane0.npy',allow_pickle=True) #Load suite2p ROI masks
print(f"Number of detected neurons: {len(stat)}") #Adding this line to print the number of detected neurons
centroids = np.array([cell['med'][::-1] for cell in stat if 'med' in cell]) #Extract neuron centroids and convert them to Numpy array
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
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', s=10, label="Suite2P Neurons")
plt.title("Mean Projection with Neurons (Plane 0)")
plt.legend()
plt.colorbar()
plt.show()





