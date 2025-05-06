#I will try to run plane 1 with iscell without projecting it to see if this will somehow explain the existence of those false positives#


import numpy as np
import matplotlib.pyplot as plt
ops=np.load('ops_plane1.npy',allow_pickle=True).item() #Load ops.npy to get metadata#
stat=np.load('stat_plane1.npy',allow_pickle=True) #Load suite2p ROI masks
iscell=np.load('iscell_plane1.npy',allow_pickle=True) #Adding iscell to tell if the detected ROI is an actual neuron
print(f"The number of detected neurons is: {len(stat)}") #Adding this line to print the number of detected neurons
accepted_neurons = [stat[i]['med'][::-1] for i in range(len(stat)) if iscell[i, 0] == 1]  # Filter only accepted neurons and swap (y,x) to (x,y) see notes
print(f"The number of accepted neurons is :{len(accepted_neurons)}")
centroids=np.array(accepted_neurons) #Convert to numpy array
Ly,Lx=ops['Ly'],ops['Lx'] #Image dimensions
n_frames=ops['nframes']   #Number of frames

#Load the binary file for plane 0#
data=np.memmap('data.bin',dtype='int16',mode='r',shape=(n_frames,Ly,Lx))

frame_index=n_frames//2 #Choosing the middle frame in order to apply the masks on one frame only

#Plot the plane with no ROI's in it for reference

plt.subplot(1,2,1)
plt.imshow(data[frame_index],cmap='gray')
plt.title('Single Frame with no ROIs overlayed (Plane 1)')
plt.legend()
plt.colorbar()


#Plot the plane with the ROI's on it#


plt.subplot(1,2,2)
plt.imshow(data[frame_index],cmap='gray') #Plotting data file of the middle frame, in grey scale
# Overlay ROI shapes
for i in range(len(stat)):
    if iscell[i, 0] == 1:
        xpix = stat[i]['xpix']
        ypix = stat[i]['ypix']
        plt.scatter(xpix, ypix, s=1, color='red', alpha=0.5)

plt.title('Single Frame with ROI Shapes Overlaid (Plane 1)')
plt.legend()
plt.colorbar()
plt.show()








