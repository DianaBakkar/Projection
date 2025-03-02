##################First attempt to make projections of 2 image stack into a coordinate system############
###############Reconstruct the image stack from data.bin file###########
##########suite2p saves the coordinates in (y,x) format but matplotlib expects them in (x,y) format thus we swap them########
import numpy as np

#Load ops.npy to get metadata for plane 0#
ops=np.load('ops.npy',allow_pickle=True).item()
stat=np.load('stat_plane0.npy',allow_pickle=True) #Load suite2p ROI masks
iscell=np.load('iscell_plane0.npy',allow_pickle=True)
print(f"Number of detected neurons: {len(stat)}") #Adding this line to print the number of detected neurons
accepted_neurons_plane0=[stat[i]['med'][::-1] for i in range(len(stat)) if iscell[i,0]==1] # Filter only accepted neurons and swap (y,x) to (x,y) see notes
centroids_plane0 = np.array(accepted_neurons_plane0)
print(f"Accepted neurons: {len(accepted_neurons_plane0)}")
Ly,Lx=ops['Ly'],ops['Lx'] #Image dimensions
n_frames=ops['nframes']   #Number of frames

#Load the binary file for plane 0#
data=np.memmap('data.bin',dtype='int16',mode='r',shape=(n_frames,Ly,Lx))

#Generate projections for plane 0#
mean_projection_plane0=np.mean(data,axis=0) #Averages pixel intensity over all frames (smooth representation)
max_projection_plane0=np.max(data,axis=0)   #Displays highest intensity per pixel (enhances bright structures)

#Save projections as .npy files
np.save('mean_projection.npy',mean_projection_plane0)
np.save('max_projection.npy',max_projection_plane0)

#Display the mean projections#
#plane 0#
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.imshow(mean_projection_plane0,cmap='gray') #Display the mean projection as a graysacle image
plt.scatter(centroids_plane0[:, 0], centroids_plane0[:, 1], color='red', s=10, label="Suite2P Neurons")
plt.title("Mean Projection with Neurons (Plane 0)")
plt.legend()
plt.colorbar()





#Load ops.npy to get metadata for plane 1#
ops=np.load('ops.npy',allow_pickle=True).item()
stat=np.load('stat_plane1.npy',allow_pickle=True) #Load suite2p ROI masks
iscell=np.load('iscell_plane1.npy',allow_pickle=True)
print(f"Number of detected neurons for plane 1: {len(stat)}") #Adding this line to print the number of detected neurons
accepted_neurons_plane1=[stat[i]['med'][::-1]for i in range(len(stat)) if iscell[i,0]==1] # Filter only accepted neurons and swap (y,x) to (x,y) see notes
print(f"Accepted neurons for plane 1: {len(accepted_neurons_plane1)}")
centroids_plane1 = np.array(accepted_neurons_plane1) #convert to numpy array
Ly,Lx=ops['Ly'],ops['Lx'] #Image dimensions
n_frames=ops['nframes']   #Number of frames

#Load the binary file for plane 1#
data_plane1=np.memmap('data_plane1.bin',dtype='int16',mode='r',shape=(n_frames,Ly,Lx))

#Generate projections for plane 1#
mean_projection_plane1=np.mean(data_plane1,axis=0) #Averages pixel intensity over all frames (smooth representation)
max_projection_plane1=np.max(data_plane1,axis=0)   #Displays highest intensity per pixel (enhances bright structures)

#Save projections as .npy files
np.save('mean_projection_plane1.npy',mean_projection_plane1)
np.save('max_projection_plane1.npy',max_projection_plane1)

#Display the mean projections#
#plane 1#
plt.subplot(1,2,2)
plt.imshow(mean_projection_plane1,cmap='gray') #Display the mean projection as a graysacle image
plt.scatter(centroids_plane1[:, 0], centroids_plane1[:, 1], color='red', s=10, label="Suite2P Neurons")
plt.title("Mean Projection with Neurons (Plane 1)")
plt.legend()
plt.colorbar()
plt.show()






