# ETH - Institute of Neuroinformatics, Neurotechnology Group.
"""
Helper Functions for:
    Algorithm Designed for tracking Spike Clusters among sessions
    
Author: Samuel Ruipérez-Campillo
"""
import numpy                   as np
import scipy                   as sp
import matplotlib.pyplot       as plt
import seaborn                 as sns
import scipy.cluster.hierarchy as shc
import mat73
import gc

from mpl_toolkits          import mplot3d
from heatmap               import heatmap, corrplot
from matplotlib            import cm
from matplotlib.colors     import ListedColormap, LinearSegmentedColormap
from sklearn.decomposition import PCA


##############################################################################
##############################################################################
## --------------------------- EXTRACT DATA ------------------------------- ##
##############################################################################
##############################################################################

def GetSessions(Data_0):
    '''
    This function takes as input the original structured file and breaks it 
    down to the third layer where the data containing the clusters (i.e. 
    waveforms and spike_times) is found. That is retrieved so that any session 
    can be accessed and its clusters inside. Also other labels are retrieved
    in case of being useful for other purposes.

    Parameters
    ----------
    Data_0 : hdf5 data file
        Contains all the sessions, clusters, and other kind of information.

    Returns
    -------
    Data_3   : hdf5 data file
        Contains all the sessions and clusters for each session. Check the main
        function to see how to extract the information from this structure.
        
    Labels_3 : string (list of lists of lists...)
        Contains lists of labels for each of the clusters inside any session.
        See main function to observe the structure of lists.

    '''
        # Layer 0. All sessions, at different dates.
    Labels_0 = list(Data_0.keys()) # Group with all the clusters
    
        # Layer 1. Layer with the dates of each experiment
    Data_1 = []
    for i in range(len(Labels_0)):
        Data_1.append(Data_0[Labels_0[i]])
    
    
    Labels_1 = []
    for i in range(len(Data_1)):
        Labels_1.append(list(Data_1[i].keys()))
    
        # Layer 2. Layers with the groups of experiments in each date - generally 'group 0'.
    Data_2 = []
    for i in range(len(Labels_1)):
        Data_22 = []
        for j in range(len(Labels_1[i])):
            Data_22.append(Data_1[i][Labels_1[i][j]])
        Data_2.append(Data_22)
    
    Labels_2 = []
    for i in range(len(Data_2)):
        Labels_22 = []
        for j in range(len(Data_2[i])):
            Labels_22.append(list(Data_2[i][j].keys()))
        Labels_2.append(Labels_22)
    
        # Layer 3- Generally, layer with the clusters of each session.
    Data_3 = []
    for i in range(len(Labels_2)):
        Data_33 = []
        for j in range(len(Labels_2[i])):
            Data_333 = []
            for k in range(len(Labels_2[i][j])):
                Data_333.append(Data_2[i][j][Labels_2[i][j][k]])
            Data_33.append(Data_333)
        Data_3.append(Data_33)
    
    Labels_3 = []
    for i in range(len(Data_3)):
        Labels_33 = []
        for j in range(len(Data_3[i])):
            Labels_333 = []
            for k in range(len(Data_3[i][j])):
                if "<class 'h5py._hl.group.Group'>" == str(type(Data_3[i][j][k])):
                    Labels_333.append(list(Data_3[i][j][k].keys()))
            Labels_33.append(Labels_333)
        Labels_3.append(Labels_33)
    Labels = [Labels_1, Labels_2, Labels_3]
    
    return Data_3, Labels


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################


##############################################################################
##############################################################################
## --------------------- ANALYTIC FUNCTIONS ------------------------------- ##
##############################################################################
##############################################################################  

def Extract_Param(sessions, Data_3, Labels_3, Sampling): 
    '''
    From the information extracted form the structured files (the lists of
    lists containing the Waveform, Spike Times and Labels), and from the 
    information about the sessions that want to be extracted from those files,
    Variables containing the Waveforms (and average waveforms for each cluster) 
    and Spike Times as well as cluster names are created. A reference time
    variable is also computed, using the input desired sampling. 

    Parameters
    ---------------------------------------------------------------------------
    sessions : array of int
        Indexes (in Data_3 and Labels_3) of the sessions from which the 
        waveforms want to be extracted.
        
    Data_3   : List of lists of lists ... of structures
        Contains all the data of waves and spike times, but needs to be 
        retrieved appropriately with the correct indexing and Labels.
        
    Labels_3 : Lists of lists of lists... of ints (or rarely strings)
        Contain in the last step the list with the number of the clusters
        that will be extracted from the appropriate session. It is used to 
        retrieve this data from Data_3.
        
    Sampling : int
        States one out of how many samples we want to use in the waveforms. It
        is important for the later dimensionality reduction, as each sample 
        is considered a 'parameter' or dimension. Thus, this Sampling variable
        determines the dimensions of our original space, where the observations
        (neuron spikes or waveforms) will be represented.

    Returns
    ---------------------------------------------------------------------------
    Waveforms    : List with x elements, of dimensions [n,m,s]
        List with all waveforms for each of the clusters:
        x : number of clusters
        n : number of spikes recorded for each of the clusters (samples)
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    av_Waveforms : List with x elements, of dimensions [m,s]
        Average waveforms for each of the clusters in all the clusters:
        x : number of clusters
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    Spike_Times : List of arrays of x floats 
        Contains the times (counted in samples, not in sec) when the spikes
        happened (x is the same as in Waveforms and av_Waveforms).
        
    clust_names : list of strings (that contain numbers)
        Strings with the name (identifier) of the clusters in the different 
        sessions
        
    Time        : Array of floats
        Uniformly distributed vector in a range of 2ms to serve as a reference
        for future plots. Its dimensions is the number of samples per spike
        divided by the Sampling variable. 

    '''
    Waveforms   = []  # Waveforms. 3D: [Spike_Times, N_Channels, Samples]
    Spike_Times = []  # Time moments (sec) at which spikes happen
    clust_names = []  # List of lists with the numbers of the clusters - one list per session.
    
    for i in range(len(sessions)): # For every desired session
        temp_clnames = []
        for j in range(len(Labels_3[sessions[i]][0][0])): # For every cluster in one session
            Waveforms.append(np.array(  Data_3[sessions[i]][0][0][Labels_3[sessions[i]][0][0][j]]['waveforms'][:,:,0::Sampling]))
            Spike_Times.append(np.array(Data_3[sessions[i]][0][0][Labels_3[sessions[i]][0][0][j]]['spike_times']))
            temp_clnames.append(Labels_3[sessions[i]][0][0][j])
        clust_names.append(temp_clnames) 

    Time = np.linspace(0,2,np.int(60/Sampling)) # Time Reference for all the spikes (samples to time conversion).
    
    av_Waveforms = [] # Average waveform of all the spikes from each cluster.
    for i in range(len(Spike_Times)):
        av_Waveforms.append(Waveforms[i].sum(axis = 0)/Waveforms[0].shape[0])
    
    return Waveforms, av_Waveforms, Spike_Times, clust_names, Time


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################

def Extract_Param_v2(directories, Sampling):
    
    Data = []
    for i in range(len(directories)):
        if directories[i][-4:] == '.mat':
            Data.append(mat73.loadmat(directories[i]))
            print('Extracting Parameters v2: Directory ' + str(i))
            
    Labels = list(Data[0].keys())
    
    D_Data = []
    d_Data = []
    
    #################################################
    
    for i in range(len(Labels)):
        d_Data = []
        for j in range(len(Data)):
            d_Data.append(Data[j][Labels[i]])
        D_Data.append(d_Data)
    
    clusterNotes        = D_Data[0]
    cluster_sites_full  = D_Data[1]
    sortedSpikeClusters = D_Data[2]
    sortedSpikes        = D_Data[3]
    spikeTimeSorted     = D_Data[4]
    spikesFiltFull      = D_Data[5]
    
    del Data
    del D_Data
    del d_Data
    gc.collect()
    
    Waveformscl = []
    clust_n   = []
    A = range(len(spikesFiltFull))
    for i in A:
        Waveforms_session = []
        ind = [] # Indexes labelled as single clusters
        for j in range(len(clusterNotes[i])):
            if clusterNotes[i][j][0] == 'single':
                ind.append(j) 
        
        indclust = [] # Indexes of the waveform for each single cluster
        for j in range(len(ind)):
            indclust.append(np.where(sortedSpikeClusters[i] == ind[j])[0])
        
        Wav = np.moveaxis(np.moveaxis(spikesFiltFull[0],0,2),0,1)
        Wav = Wav[:,:,0:Wav.shape[2]-1] # Remove last sample to have 60, not 61        
        
            # Remove Unnecesary parts of the huge variable of Waveforms
        Temp_Spikes = spikesFiltFull[1:] # Temporal Variable wo\ used section
        del spikesFiltFull
        spikesFiltFull = Temp_Spikes     # New version occupying less memory
        del Temp_Spikes
        gc.collect()
        
        for j in range(len(ind)): # For each cluster
            Wav_clust = []
            for k in indclust[j]:
                Wav_clust.append(Wav[k,:,0::Sampling])
            Waveforms_session.append(np.asarray(Wav_clust))   
        
        Waveformscl.append(Waveforms_session)
        clust_n.append(len(ind)) # List of number of clusters per session
        print('Extracting Waveforms: Cluster' + str(i))
        
    Waveforms = []
    for i in range(len(Waveformscl)):
        Waveforms = Waveforms + Waveformscl[i]
    
    Spike_Times = spikeTimeSorted  # Time moments (sec) at which spikes happen
    
    clust_names = []
    for i in range(len(directories)):
        if directories[i][-4:] == '.mat':
            clust_names.append(directories[i][-10:-3])  # List of lists with the numbers of the clusters - one list per session.
            print(i)
            
    av_Waveforms = [] # Average waveform of all the spikes from each cluster.
    for i in range(len(Waveforms)):
        av_Waveforms.append(Waveforms[i].sum(axis = 0)/Waveforms[0].shape[0])
        
        
    Time = np.linspace(0,2,np.int(60/Sampling)) # Time Reference for all the spikes (samples to time conversion).

    return Waveforms, av_Waveforms, Spike_Times, clust_names, Time, clust_n


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
    
def extract_HAwav_t(Option, n_chan, Waveforms, av_Waveforms, Clust_ref):
    '''
    From the average waveforms of all the channels of all the clusters, this 
    function extracts the highest amplitude n channels (spikes) of a reference 
    cluster and the spike waves in these channels for the other clusters (i.e.
    spatial information is maintained as same channels are selected for all 
    clusters). It also returns the ordered indices of such channels.

    Parameters
    ---------------------------------------------------------------------------
    n_chan       : int
        Number of channels that are going to be concatenated.
        
    Waveforms    : List with x elements, of dimensions [n,m,s]
        List with all waveforms for each of the clusters:
        x : number of clusters
        n : number of spikes recorded for each of the clusters (samples)
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    av_Waveforms : List with x elements, of dimensions [m,s]
        Average waveforms for each of the clusters in all the clusters:
        x : number of clusters
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    Clust_ref    : int
        Cluster taken as a reference to decide the number of channels with
        highest Amplitude.
    
    Returns
    ---------------------------------------------------------------------------
    Waveforms_HA_c : list of x elements, of dimensions [p,q]
        Contains the concatenation of the 5 channels selected, for each of the
        clusters (e.g. in the case of the reference cluster, the channels where
        the highest amplitude spikes are found).
    
    indices_maxn   : List of 1 element with an array of n elements
        Contains the indices with the maximum amplitude of the reference. n is
        the number of indices (i.e. n_chan).
    '''
    
    n = n_chan         # Number of indexes that want to be saved
    Waveforms_sum = [] # List with the components of the absolute sum of each av. spike channel for reference cluster.
    indices_maxn  = [] # List of channels ordered by their highest average spike amplitude (dicrete integral).
    Clust_ref     = 0  # Select the reference cluster (with respect to the temporal aligning)
    Channels_sum  = []
    for j in range(av_Waveforms[Clust_ref].shape[0]): # For each Channel
        Channels_sum.append(np.sum(np.abs(av_Waveforms[Clust_ref][j]))) # Append the abs sum of each channel.  
    
    Channels_sum = np.array(Channels_sum)
    idx_maxn = [] 
    idx_maxn = np.argpartition(Channels_sum, -n)[-n:] # indexes of max amplitude - ordered (smallest A. - biggest A.)
    
    indices_maxn.append(idx_maxn[np.argsort((-Channels_sum)[idx_maxn])])  # Append channel index w\ highest spike amplitude.
    Waveforms_sum.append(Channels_sum) # Append the abs sum of all the channels for each cluster.
    
        # Concatenate spikes (samples) of channels with highest amplitude.
    Waveforms_HA   = [] # List of lists with samples from n spikes (temorally aligned to HA of Clust_ref)
    Waveforms_HA_c = [] # Concatenated n channels of waveforms
    for j in range(len(Waveforms)): # For every Cluster
        Waveforms_HA_channel = []   # List of n temporally aligned spikes for channel j
        for i in range(n):          # For every Channel n
            Waveforms_HA_channel.append(Waveforms[j][:,indices_maxn[0][i],:])
        Waveforms_HA.append(Waveforms_HA_channel)
        Waveforms_HA_c.append(np.concatenate(Waveforms_HA[j], axis = 1)) # Concatenated n channels of waveforms
    
    return Waveforms_HA_c, indices_maxn


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################

def extract_HAwav(n_chan, Waveforms, av_Waveforms):
    '''
    From the average waveforms of all the channels of all the clusters, this 
    function extracts the highest amplitude n channels (spikes) of all the 
    clusters (i.e. the spatial information is NOT maintained as NOT same 
    channels are selected for all clusters). It also returns the ordered 
    indices of such channels.

    Parameters
    ---------------------------------------------------------------------------
    n_chan       : int
        Number of channels that are going to be concatenated.
        
    Waveforms    : List with x elements, of dimensions [n,m,s]
        List with all waveforms for each of the clusters:
        x : number of clusters
        n : number of spikes recorded for each of the clusters (samples)
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    av_Waveforms : List with x elements, of dimensions [m,s]
        Average waveforms for each of the clusters in all the clusters:
        x : number of clusters
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    
    Returns
    ---------------------------------------------------------------------------
    Waveforms_HA_c : list of x elements, of dimensions [p,q]
        Contains the concatenation of the 5 channels selected, for each of the
        clusters (e.g. in all cases, the channels where the highest amplitude 
        spikes are found).
        
    indices_maxn   : List of 1 element with an array of n elements
        Contains the indices with the maximum amplitude of the reference. n is
        the number of indices (i.e. n_chan)
    '''  
    
    n = n_chan          # Number of indexes that want to be saved
    Waveforms_sum = [] # List with the components of the absolute sum of each av. spike channel for each cluster.
    indices_maxn  = [] # List of channels ordered by their highest average spike amplitude (dicrete integral).
    for i in range(len(av_Waveforms)): # For each Cluster
        Channels_sum = []
        for j in range(av_Waveforms[i].shape[0]): # For each Channel
            Channels_sum.append(np.sum(np.abs(av_Waveforms[i][j]))) # Append the abs sum of each channel.  
        
        Channels_sum = np.array(Channels_sum)
        idx_maxn = [] 
        idx_maxn = np.argpartition(Channels_sum, -n)[-n:]
        
        indices_maxn.append(idx_maxn[np.argsort((-Channels_sum)[idx_maxn])])  # Append channel index w\ highest spike amplitude.
        Waveforms_sum.append(Channels_sum) # Append the abs sum of all the channels for each cluster.
    
    # Concatenate spikes (samples) of channels with highest amplitude.
    Waveforms_HA   = [] # List of lists with samples from n highest amplitude spikes
    Waveforms_HA_c = [] # Concatenated n channels of waveforms
    for j in range(len(Waveforms)): # For every Cluster
        Waveforms_HA_channel = [] # List of n highest amplitude spikes for channel j
        for i in range(n): # For every Channel n
            Waveforms_HA_channel.append(Waveforms[j][:,indices_maxn[j][i],:])
        Waveforms_HA.append(Waveforms_HA_channel)
        Waveforms_HA_c.append(np.concatenate(Waveforms_HA[j], axis = 1)) # Concatenated n channels of waveforms
    
    return Waveforms_HA_c, indices_maxn


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
    
def extract_HAwav_av(Option, n_chan, Waveforms_HA_c):
    '''
    In case of working on the modes Option 2 or 3, it returns the average 
    waveform of the concatenated channels of higher amplitude.

    Parameters
    ---------------------------------------------------------------------------
    Option         : int (1, 2 or 3)
        Mode of operation
        
    n_chan         : int
        Number of channels of the recording system (i.e. electrodes)
        
    Waveforms_HA_c : list of x elements, of dimensions [p,q]
        Contains the concatenation of the 5 channels selected, for each of the
        clusters (e.g. in all cases, the channels where the highest amplitude 
        spikes are found).
    
    Returns
    ---------------------------------------------------------------------------
    Waveforms_HA_c_av : list of x elements, of dimensions [q,]
        Contains the average value of the concatenation of all the channels 
        selected for each of the clusters (the average is taken from all the
        observations of spike waveforms)
        q : number of dimensions: samples of concatenated spikes (60 x n, with 
        n as number of waves/channels)
    '''  
    
    if Option == 2 or Option == 3:
        Waveforms_HA_c_av = []
        # Time_2 = Time = np.linspace(0,2*n_chan,Waveforms_HA_c[0].shape[1])
        for i in range(len(Waveforms_HA_c)):
            Waveforms_HA_c_av.append(np.mean(Waveforms_HA_c[i], axis = 0)) # Av Waveform for each cluster.
    
    return Waveforms_HA_c_av


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################

def extract_Allwav(Waveforms):
    '''
    From the average waveforms of all the channels of all the clusters, this 
    function extracts all the channels from all the clusters (i.e. the spatial 
    information is maintained as all clusters as selected). 

    Parameters
    ---------------------------------------------------------------------------
        
    Waveforms    : List with x elements, of dimensions [n,m,s]
        List with all waveforms for each of the clusters:
        x : number of clusters
        n : number of spikes recorded for each of the clusters (samples)
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    av_Waveforms : List with x elements, of dimensions [m,s]
        Average waveforms for each of the clusters in all the clusters:
        x : number of clusters
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    
    Returns
    ---------------------------------------------------------------------------
    Waveforms_HA_c : list of x elements, of dimensions [p,q]
        Contains the concatenation of the 5 channels selected, for each of the
        clusters (e.g. in all cases, the channels where the highest amplitude 
        spikes are found).
        
    '''  
    
    Waveforms_HA_c = []
    for i in range(len(Waveforms)): # For each cluster of spikes
        Wav_ind = []
        Waveforms_chan = [] # All channels together concatenated for each spike
        Wav_ind = np.stack(Waveforms[i], axis = 2) # Change dim. order
        Waveforms_chan = np.concatenate(Wav_ind, axis = 0)
        Waveforms_HA_c.append(Waveforms_chan.transpose())

    return Waveforms_HA_c


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
    
def conc_Allsamp(Waveforms_HA_c):
    '''
    From the concatenated waveforms for each spike for every cluster, all
    spikes are joint in a variable in a n-Dimensional space (with n number of
    sample points in each of the concatenated waveform)

    Parameters
    ---------------------------------------------------------------------------
        
    Waveforms_HA_c : list of x elements, of dimensions [p,q]
        Contains the concatenation of the all channels, for each of the
        clusters.
        p : number of spikes for each of the clusters (different in each case).
        q : number of dimensions: samples of concatenated spikes (60 x n, with 
        n as number of waves/channels)
        
    Returns
    ---------------------------------------------------------------------------
    Waveforms_Allsamp : list of x elements, of dimensions [p,q]
        Contains the concatenation of the 5 channels selected, for each of the
        clusters (e.g. in all cases, the channels where the highest amplitude 
        spikes are found).
        
    '''  
    
    Conc = Waveforms_HA_c[0]
    for i in range(len(Waveforms_HA_c)-1):
        Conc = np.concatenate((Conc, Waveforms_HA_c[i+1]), axis = 0)
    Waveforms_Allsamp = Conc
    
    return Waveforms_Allsamp


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
    
def compute_distmat(Proj, metric):
    '''
    From the Projection matrices, that is, the matrices containing the points
    projected into the Principal-Components Subspace previously designed, the 
    distance in such subspace is measured between clusters belonging to 
    different spike clusters.

    Parameters
    ---------------------------------------------------------------------------
    Proj    : List of x elements with dimensions [m,n]
        List of the projections matrices. Each element x_i of the list contains
        one matrix representing the clusters of all samples (spikes) 
        projected (reduced) to the Principal-Components subspace. The dimensions
        of each of these matrices is [m,n]:
        m : number of dimensions for each sample (spike), i.e. number of PC axes
        n : number of samples (spikes) per cluster
        
    metric  : String
        String containing the metric. We assume so far that it is mahalanobis.
    
    Returns
    ---------------------------------------------------------------------------
    Av_dist_mat : np.array matrix of dimensions [q,q]
        Contains the average (mahalanobis) distance between clusters. The 
        diagonal represents the distance of one cluster with itself, and the
        rest of columns, the distance of such cluster to all the other clusters.
        q : number of clusters (of spikes).
    
    Side Notes
    ---------------------------------------------------------------------------
    Important Note I: Note that the distance of each cluster with itself is not
        zero. This is because the distance is computed as the average distance 
        of each point to the rest of points. Therefore, one point in its cluster
        is nearer to the rest of points of the cluster, so the distance is 
        shorter but not zero.
        
    Important Note II: In order to compute the distance of a cluster to itself, 
        LOO (Leave-one-out) algorithm is utilised, meaning that the point being
        evaluated is taken out of the distribution of its own cluster, when
        measuring the mahalanobis distance of such point to the rest of the 
        points of its cluster. Thence, biases are avoided.
    '''  
    
    Av_dist = [] # List of average distance from every point in cluster j to every cluster i. Diagonal!!!       
    
    for j in range(len(Proj)): # For every cluster j
        Point_mahalab = [] # List of lists with distances. For every point k, distances to every cluster i.
        Distances_singleclust = np.zeros((Proj[j].shape[1], len(Proj))) # Matrix of distances of every point in cluster j
                                                                        # to every cluster distribution.
        for k in range(Proj[j].shape[1]): # And for every point k in cluster j
            Test = Proj[j][:,k] # Point k in cluster j, to be measured to every cluster i.
            Test = np.transpose(np.expand_dims(Test, axis = 1))
            Mahalab_dist  = [] # Distance of point k to every other cluster i.
            for i in range(len(Proj)): # For every cluster i           
                if i == j: # Exclude Test points from 'Training'. Avoid biases.
                    Data_c = np.transpose(np.delete(Proj[i],k,1)) # Delete kth point of the cluster
                else:
                    Data_c = np.transpose(Proj[i])
                Data = Data_c
                Mahalab_dist.append(np.mean(sp.spatial.distance.cdist(Test, Data, metric)))
            Point_mahalab.append(Mahalab_dist)
        for l in range(len(Point_mahalab)): # Transform List of lists Point_mahalab into a matrix
                                            # of dimensions [m x n], m: # points in cluster j; n: # clusters
            Distances_singleclust[l,:] = np.squeeze(np.array(Point_mahalab[l]))
        Av_dist.append(np.mean(Distances_singleclust, axis = 0)) # Matrix of average distances from points in cluster j to every other cluster.
                                                                 # Note that the diagonal is av. dist. to its own cluster!!
        print('Compute MD Cluster' + str(j))
    Av_dist_mat = np.zeros((len(Proj),len(Proj)))   
    for l in range(Av_dist_mat.shape[0]): # Transform List of lists Point_mahalab into a matrix
                                          # of dimensions [m x n], m: # points in cluster j; n: # clusters
        Av_dist_mat[l,:] = np.squeeze(np.array(Av_dist[l]))
    # Av_dist_mat = Av_dist_mat/(Av_dist_mat.max())
    return Av_dist_mat


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
    
def PCA_dimred(n_pca, perc_pca, Suboption, Waveforms_ref):
    '''
    Create a Principal-Components subspace from the original space where the
    waveforms of the spikes were represented. This subspace will be used to 
    represent them in a lower dimensional framework.

    Parameters
    ---------------------------------------------------------------------------
    n_pca         : int
        Number of Principal-Components (that is, axis in our projection sub-
        space).
    perc_pca      : float
        Percentage of variance maintained in the PC-subspace. Also understood
        as information or energy (in [0,1])
    Suboption     : int
        0 or 1 depending on if the n_pca is taken into account (0) or perc_pca(1)
    Waveforms_ref : array of floats, dimensions n x m
        This array contains concatenated all the samples from the original 
        space in the original dimensions, i.e. the samples in the space from
        which the PC-subspace is formed.
        n: number of samples or spikes.
        m: number of original dimensions.
        
    Returns
    ---------------------------------------------------------------------------
    pca_subspace : PCA Object (decomposition._pca.PCA)
        Object containing the information about the new PC-subspace after 
        having reduced dimensionality.

    '''
    
    if Suboption == 0:
        pca_subspace = PCA(n_components = n_pca)  
    elif Suboption == 1:
        pca_subspace = PCA(n_components = perc_pca, svd_solver = 'auto') # Pick the ammount of variance we want to see (e.g. 75)
    pca_subspace.fit(Waveforms_ref)
    
    return pca_subspace


##############################################################################
##############################################################################
## ------------------------ DISPLAY FUNCTIONS ----------------------------- ##
##############################################################################
##############################################################################
    
def PlotSingleWaveforms(Waveforms, av_Waveforms, Time, cl):
    '''
    Plot a set of Waveform spikes in different figures

    Parameters
    ---------------------------------------------------------------------------
    Waveforms    : List with x elements, of dimensions [n,m,s]
        List with all waveforms for each of the clusters:
        x : number of clusters
        n : number of spikes recorded for each of the clusters (samples)
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    av_Waveforms : List with x elements, of dimensions [m,s]
        Average waveforms for each of the clusters in all the clusters:
        x : number of clusters
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    Time        : Array of floats
        Uniformly distributed vector in a range of 2ms to serve as a reference
        for future plots. Its dimensions is the number of samples per spike
        divided by the Sampling variable.
    cl           : int
        Cluster that is meant to be used for the plots
        
    Returns
    ---------------------------------------------------------------------------
    No variable returned: A plot of the Waveforms is the output.
    '''
    
    font = {'family' : 'Times New Roman',
                'weight' : 'bold',
                'size'   : 15}
    
    for i in range(Waveforms[cl].shape[1]):
        fig, ax = plt.subplots()
        namefig = 'Cluster ' + str(cl) + ' channel ' + str(i) +'.pdf'
        namefig2 = 'Cluster ' + str(cl) + ' channel ' + str(i) +'.svg'
        plt.xlabel('Time (sec)', fontfamily = 'Times New Roman', fontsize = 18)
        plt.ylabel('Amplitude (mV)', fontfamily = 'Times New Roman', fontsize = 18)
        plt.title(namefig[0:-4], fontfamily = 'Times New Roman', fontsize = 18)
        for j in range(Waveforms[cl].shape[0]):
            plt.plot(Time,Waveforms[cl][j,i,:],c ='darkgrey', linewidth = 0.1)
        plt.plot(Time,av_Waveforms[cl][i,:],   c = 'black',   linewidth = 2)
        ax.set(xlim=(Time[0], Time [-1]))
        plt.rc('font', **font)
        plt.grid(b=bool, which='major', axis='both', color='gray', linestyle='--', linewidth=0.6)
        plt.minorticks_on()
        plt.grid(b=bool, which='minor', axis='both', color='silver', linestyle='--', linewidth=0.3)
        
        plt.savefig(namefig)
        plt.savefig(namefig2)


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
        
def PlotWaveforms(Print_S_Opt, Option, Waveforms_HA_c, Waveforms_HA_c_av, Time_2, n_chan, indices_maxn):
    '''
    This function concatenates the spike waves from all the channels taken into
    account from each of the clusters (i.e. for each of the neuronal activities),
    and plot different figures with that information.

    Parameters
    ---------------------------------------------------------------------------
    Print_S_Opt       : int (2 or 3)
        Printing mode - All clusters separately and together (2) or only one 
        figure with all clusters together (3).
    Option            : int (1, 2 or 3)
        Mode of operation
        
    Waveforms_HA_c    : list of x elements, of dimensions [p,q]
        Contains the concatenation of the all channels, for each of the
        clusters.
        p : number of spikes for each of the clusters (different in each case).
        q : number of dimensions: samples of concatenated spikes (60 x n, with 
        n as number of waves/channels)
        
    Waveforms_HA_c_av : list of x elements, of dimensions [q,]
        Contains the average value of the concatenation of all the channels 
        selected for each of the clusters (the average is taken from all the
        observations of spike waveforms)
        q : number of dimensions: samples of concatenated spikes (60 x n, with 
        n as number of waves/channels)
        
    Time              : Array of floats
        Uniformly distributed vector in a range of 2ms to serve as a reference
        for future plots. Its dimensions is the number of samples per spike
        divided by the Sampling variable. 
        
    n_chan            : int
        Number of channels that are going to be concatenated.
        
    indices_maxn      : List of 1 element with an array of n elements
        Contains the indices with the maximum amplitude of the reference. n is
        the number of indices (i.e. n_chan)

    Returns
    ---------------------------------------------------------------------------
    No variable returned: A plot of the concatenated waveforms for all clusters
    is the output.

    ''' 
           
        
    font = {'family' : 'Times New Roman',
                'weight' : 'bold',
                'size'   : 15}
    
    if Print_S_Opt == 2: # Plot every single concatenation of spikes
        for i in range(len(Waveforms_HA_c)):
            f, ax = plt.subplots()
            plt.suptitle('Temporary aligned HA Spikes', fontfamily = 'Times New Roman', fontsize = 22)
            plt.xlabel('Concatenated Channels (Time in sec - reference)', fontfamily = 'Times New Roman', fontsize = 18)
            plt.ylabel('Amplitude (mV)', fontfamily = 'Times New Roman', fontsize = 18)
            
            samp = Waveforms_HA_c[i].shape[0]//200 + 1
            rep  = np.linspace(0,Waveforms_HA_c[i].shape[0],Waveforms_HA_c[i].shape[0]//samp)
            rep_int = [np.int(x) for x in rep]
            
            for j in rep_int[:-1]:
                plt.plot(Time_2,Waveforms_HA_c[i][j,:], c ='darkgrey', linewidth = 0.1)
            plt.plot(Time_2,Waveforms_HA_c_av[i],       c = 'black',   linewidth = 2)
            
            for xc in range(n_chan):
                if Option == 2:
                    plt.axvline(x = xc*2, c = 'darkgrey' , ls = ':', lw = 2, linewidth = 1.5, label = 'Channel number: ' + str(indices_maxn[0][xc]))
                elif Option == 3: 
                    plt.axvline(x = xc*2, c = 'darkgrey' , ls = ':', lw = 2, linewidth = 1.5)
            plt.axvline(x = xc*2+2, c = 'darkgrey' , ls = ':', lw = 2, linewidth = 1.5)
            if Option == 2:
                plt.legend(loc = 'lower right')
            namefig = 'Cluster_' + str(i) + '.pdf'
            namefig2 = 'Cluster_' + str(i) + '.svg'
            
            # ax.set(xlim=(Time[0], Time [-1]))
            plt.rc('font', **font)
            plt.grid(b=bool, which='major', axis='both', color='gray', linestyle='--', linewidth=0.6)
            plt.minorticks_on()
            plt.grid(b=bool, which='minor', axis='both', color='silver', linestyle='--', linewidth=0.3)
            
            plt.savefig(namefig)
            plt.savefig(namefig2)
        
        # Plot all clusters in the same figure
    fig, axs = plt.subplots(len(Waveforms_HA_c), sharex = True, sharey = False)
    plt.suptitle( 'Temporary aligned HA Spikes', fontweight = 'heavy', fontfamily = 'Times New Roman', fontsize = 18)
    plt.xlabel('Time (sec)', fontfamily = 'Times New Roman')
    fig.text(0.04, 0.5, 'Amplitude (mV)', va='center', ha='center', rotation='vertical', fontfamily = 'Times New Roman')
    for i in range(len(Waveforms_HA_c)):
        
        samp = Waveforms_HA_c[i].shape[0]//150 + 1
        rep  = np.linspace(0,Waveforms_HA_c[i].shape[0],Waveforms_HA_c[i].shape[0]//samp)
        rep_int = [np.int(x) for x in rep]
            
        for j in rep_int[:-1]:
            axs[i].plot(Time_2,Waveforms_HA_c[i][j,:], c ='darkgrey', linewidth = 0.08)
        axs[i].plot(Time_2,Waveforms_HA_c_av[i],       c = 'black',   linewidth = 1)
        for xc in range(n_chan):
            axs[i].axvline(x = xc*2, c = 'dimgray' , ls = ':', lw = 2, linewidth = 1)
        axs[i].axvline(x = xc*2+2, c = 'dimgray' , ls = ':', lw = 2, linewidth = 1)
        
        axs[i].grid(b=bool, which='major', axis='both', color='gray', linestyle='--', linewidth=0.6)
        axs[i].minorticks_on()
        axs[i].grid(b=bool, which='minor', axis='both', color='silver', linestyle='--', linewidth=0.3)
        
        print('Plotting All Channel Spikes, cluster ' + str(i))
    f = plt.gcf()  # f = figure(n) if you know the figure number
    f.set_size_inches(8.27,11.69)
        
    if len(indices_maxn) == 0:
        t = np.linspace(0,42,43)
        t_Names = [
                    '', 'Channel 0',  '', 'Channel 1',  '', 'Channel 2', 
                    '', 'Channel 3',  '', 'Channel 4',  '', 'Channel 5',
                    '', 'Channel 6',  '', 'Channel 7',  '', 'Channel 8',
                    '', 'Channel 9',  '', 'Channel 10', '', 'Channel 11',
                    '', 'Channel 12', '', 'Channel 13', '', 'Channel 14',
                    '', 'Channel 15', '', 'Channel 16', '', 'Channel 17',
                    '', 'Channel 18', '', 'Channel 19', '', 'Channel 20', ''
                    ]

        plt.xticks(t, t_Names, rotation = 70, fontsize = 5)
    
    namefig = 'Alltogether' + '.pdf'
    namefig2 = 'Alltogether' + '.svg'
    fig.savefig(namefig)
    fig.savefig(namefig2)


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
            
def ClustPlot(n_pca, Proj):
    '''
    The current function takes the number of PCA components and
    
    Parameters
    --------------------------------------------------------------------------
    n_pca : int
        Number of Principal-Components (that is, axis in our projection sub-
        space).
    Proj  : list n elements pxq
        Represents the clusters of spikes in the projection subspace.
        n: Number of clusters that have been projected in the p-dimensional space.
        p: dimensions of the subspace.
        q: number of samples or observations (spikes) for each neuron cluster.

    Returns
    --------------------------------------------------------------------------
    ax1 : fig object
        Figure File of the 3D projection with initial view 1.
    ax2 : fig object
        Figure File of the 3D projection with initial view 2.
    ax2 : fig object
        Figure File of the 3D projection with initial view 3.

    '''
    
    cmaps = [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        'hot', 'afmhot', 'gist_heat', 'copper', 'spring', 'summer', 
        'autumn', 'winter', 'cool', 'Wistia', 'binary', 'gist_yarg', 
        'gist_gray', 'gray', 'bone', 'pink'
    ]   # List of Colour maps.
    
    font = {'family' : 'Times New Roman',
                'weight' : 'bold',
                'size'   : 15}
    
    if n_pca == 3:
            # Plot Figures in different views of the 3D projection
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.view_init(60, 65)    
        for i in range(len(Proj)):
            ax.scatter3D(Proj[i][0,:], Proj[i][1,:], Proj[i][2,:], cmap = cmaps[i], alpha = 0.3,linewidths = 0.7) 
        
    if n_pca == 2:
        fig = plt.figure()
        ax = plt.axes
        for i in range(len(Proj)):
            fig = plt.scatter(Proj[i][0,:], Proj[i][1,:], cmap = cmaps[i], alpha = 0.3,linewidths = 0.4)
        plt.grid(b=bool, which='major', axis='both', color='silver', linestyle='--', linewidth=0.6)
        # plt.grid(b=bool, which='minor', axis='both', color='silver', linestyle='--', linewidth=0.3)
        
    plt.xlabel('X', fontfamily = 'Times New Roman', size = 20)
    plt.ylabel('Y', fontfamily = 'Times New Roman', size = 20)
    Tit = 'PC-Subspace with ' + str(n_pca) + ' Components'
    plt.title(Tit, fontfamily = 'Times New Roman')
    plt.rc('font', **font)
    
    name_fig = Tit + '.pdf'
    name_fig2 = Tit + '.png'
    plt.savefig(name_fig)
    plt.savefig(name_fig2)
    
    return ax


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
    
def plotMahala_mat(Av_dist_mat, Option, Suboption, n_chan, Tot_Var):
    '''
    

    Parameters
    ---------------------------------------------------------------------------
    Av_dist_mat : np.array matrix of dimensions [q,q]
        Contains the average (mahalanobis) distance between clusters. The 
        diagonal represents the distance of one cluster with itself, and the
        rest of columns, the distance of such cluster to all the other clusters.
        q : number of clusters (of spikes).
        
    Option      : int (1, 2 or 3)
        Mode of operation
        
    Suboption   : int
        0 or 1 depending on if the n_pca is taken into account (0) or perc_pca(1)
        
    n_chan      : int
        Number of channels that are going to be concatenated.
        
    Tot_Var     : float
        Percentage of information or variance maintained in the PC-subspace.

    Returns
    ---------------------------------------------------------------------------
    None.

    '''
    
    Tit = 'Mah_Dist_w_' + 'Opt_' + str(Option) + '_Subopt_' + str(Suboption) + '_NChan_' + str(n_chan) + '_VarPCA_' + str("{:.2f}".format(Tot_Var))
    f = plt.figure()
    plt.title(Tit)
    sns.heatmap(Av_dist_mat, annot = True)
    plt.xlabel('Cluster Number')
    plt.ylabel('Cluster Number')
    name_fig = Tit + '.pdf'
    f.savefig(name_fig)


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
        
def plot_hierarchical_dend(Red_mat, hier_method):
    
    print('Plotting Hierarch. Dend.')      
    Link_mat = shc.linkage(Red_mat, method = hier_method, metric = 'euclidean', optimal_ordering = False)
    
    fig = plt.figure()
    dend = shc.dendrogram(Link_mat)
    plt.xlabel('Cluster Number', fontfamily = 'Times New Roman')
    plt.ylabel('Cluster Distance Metric', fontfamily = 'Times New Roman')
    Tit = 'Hierarchical Linkage Clustering with ' + str(hier_method) + ' Method'
    plt.title(Tit, fontfamily = 'Times New Roman')
    Link = shc.linkage(Red_mat, hier_method)
    # fig.grid(b=bool, which='minor', axis='both', color='silver', linestyle='--', linewidth=0.3)
    # minorticks_on()
    name_fig = Tit + '.pdf'
    fig.savefig(name_fig)
    
    return Link
    