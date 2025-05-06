##################First attempt to make projections of image stack into a coordinate system############
###############Reconstruct the image stack from data.bin file###########
##########suite2p saves the coordinates in (y,x) format but matplotlib expects them in (x,y) format thus we swap them########
import numpy as np

ops=np.load('ops.npy',allow_pickle=True).item() #Load ops.npy to get metadata#
stat=np.load('stat_plane0.npy',allow_pickle=True) #Load suite2p ROI masks
iscell = np.load('iscell_plane0.npy', allow_pickle=True) #Adding iscell to tell if the detected ROI is an actual neuron
print(f"Number of detected neurons: {len(stat)}") #Adding this line to print the number of detected neurons
accepted_neurons = [stat[i]['med'][::-1] for i in range(len(stat)) if iscell[i, 0] == 1]  # Filter only accepted neurons and swap (y,x) to (x,y) see notes
print(f"Number of accepted neurons : {len(accepted_neurons)}")
centroids = np.array(accepted_neurons) #Convert to numpy array
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
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', s=10, label="")
plt.title("Mean Projection with Neurons (Plane 0)")
plt.legend()
plt.colorbar()
plt.show()





