##################First attempt to make projections of 2 image stack into a coordinate system############
###############Reconstruct the image stack from data.bin file###########
import numpy as np
#Load ops.npy to get metadata for plane 0#
ops=np.load('ops.npy',allow_pickle=True).item()
stat=np.load('stat_plane0.npy',allow_pickle=True) #Load suite2p ROI masks
print(f"Number of detected neurons: {len(stat)}") #Adding this line to print the number of detected neurons
centroids = np.array([cell['med'][::-1] for cell in stat if 'med' in cell]) #Extract neuron centroids and convert them to Numpy array
Ly,Lx=ops['Ly'],ops['Lx'] #Image dimensions
n_frames=ops['nframes']   #Number of frames

#Load ops.npy to get metadata for plane 1#
ops=np.load('ops.npy',allow_pickle=True).item()
stat=np.load('stat_plane1.npy',allow_pickle=True) #Load suite2p ROI masks
print(f"Number of detected neurons: {len(stat)}") #Adding this line to print the number of detected neurons
centroids = np.array([cell['med'][::-1] for cell in stat if 'med' in cell]) #Extract neuron centroids and convert them to Numpy array
Ly,Lx=ops['Ly'],ops['Lx'] #Image dimensions
n_frames=ops['nframes']   #Number of frames


#Load the binary file for plane 0#
data=np.memmap('data.bin',dtype='int16',mode='r',shape=(n_frames,Ly,Lx))

#Load the binary file for plane 1#
data_plane1=np.memmap('data_plane1.bin',dtype='int16',mode='r',shape=(n_frames,Ly,Lx))

#Generate projections for plane 0#
mean_projection=np.mean(data,axis=0) #Averages pixel intensity over all frames (smooth representation)
max_projection=np.max(data,axis=0)   #Displays highest intensity per pixel (enhances bright structures)

#Generate projections for plane 1#
mean_projection_plane1=np.mean(data_plane1,axis=0) #Averages pixel intensity over all frames (smooth representation)
max_projection_plane1=np.max(data_plane1,axis=0)   #Displays highest intensity per pixel (enhances bright structures)

#Save projections as .npy files
np.save('mean_projection.npy',mean_projection)
np.save('max_projection.npy',max_projection)
np.save('mean_projection_plane1.npy',mean_projection_plane1)
np.save('max_projection_plane1.npy',max_projection_plane1)

#Display the mean projections#
#plane 0#
import matplotlib.pyplot as plt
plt.imshow(mean_projection,cmap='gray') #Display the mean projection as a graysacle image
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', s=10, label="Suite2P Neurons")
plt.title("Mean Projection with Neurons (Plane 0)")
plt.legend()
plt.colorbar()
plt.show()

#plane 1#
plt.imshow(mean_projection,cmap='gray') #Display the mean projection as a graysacle image
plt.scatter(centroids[:, 0], centroids[:, 1], color='blue', s=10, label="Suite2P Neurons")
plt.title("Mean Projection with Neurons (Plane 0)")
plt.legend()
plt.colorbar()
plt.show()






