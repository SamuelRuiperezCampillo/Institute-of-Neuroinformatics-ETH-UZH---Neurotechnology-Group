# ETH - Institute of Neuroinformatics, Neurotechnology Group.
"""
Algorithm Designed for tracking Spike Clusters among sessions
Author: Samuel Ruipérez-Campillo

Version 3 - Cool Results with sample clusters from first sessions. Mahalanobis
            has been implemented and also has the PCA Analysis been.
"""

get_ipython().magic('reset -sf') # Clear Workspace

import numpy             as np
import matplotlib.pyplot as plt
import h5py              as h5
import Track_Functions   as TF
import os
from scipy.spatial.distance import pdist


plt.close('all') # Clear Plots
folder = os.getcwd()
os.chdir(r'C:\Users\sruip\Desktop\Universities\ETH\Research_Projects\Lab_Institute_Fur_Neuroinformatics\mBY03_sorted_units_v2') 

# %% Extract Spikes from Structured Data file

Data_0         = h5.File('analysis_results.hdf5', 'r') 
Data_3, Labels = TF.GetSessions(Data_0)      # Check Track_Functions
Labels_3       = Labels[2]

# %% Extract Specific Sessions -  User can change these parametres, see Labels_3
    # Extract Sessions, e.g.:
        # Session 20190909 - Clusters: 10_12_16_18_25_27_4_6
        #                  - Index 10 in layer 1
        # Session 20190910 - Clusters: 10_14_15_16_23_28_29_4_40_42_52_53_63_7
        #                  - Index 11 in layer 1
sessions_n  = ['S. 20190909','S. 20190910']  # List of Names: Sessions   
sessions    = np.array([14,15])                 # Array of indexes of desired sessions
Sampling    = 1                              # The signal is sampled over this Sampling Factor 
                                             # (i.e. one out of 'Sampling' samples)

Waveforms, av_Waveforms, Spike_Times, clust_names, Time = TF.Extract_Param(sessions, Data_3, Labels_3, Sampling)
        
# %% Options for Analysis and General Data/Variables
            
Option      = 3    # Choose: 0 - Approach of PCA one channel and one cluster. Other clusters projected in that space.
                   #         1 - Approach of PCA Highest Amplitude Channels (HAC) of one cluster. Other HAC projected in that space.
                   #         2 - Approach of PCA Highest Amplitude Channels of one cluster. The same channels from other clusters, project into that space.
                   #         3 - Approach of PCA All channels from all clusters: Big Space to small PCA subspace.

Suboption   = 0    # Choose: 0 - number of components of the PCA Analysis is constant (e.g. 3)
                   #         1 - number of components of PCA depends on the amount of variance desired (e.g. 60%)

Print_S_Opt = 0    # Choose: 0 - No print
                   # Choose: 1 - plot a set of single spikes in different figures
                   # Choose: 2 - Plot concatenated waveforms: Each Single Cluster and All Together
                   # Choose: 3 - Plot concatenated waveforms: Only all Clusters Together


Option_Mah  = 1    # Choose: 0 - No computation of the metric
                   # Choose: 1 - Computation of the metric

n_pca    = 3     # Number of Principal components if Suboption = 0
perc_pca = 0.50  # Percentage of variance in pca  if Subotpion = 1
n_chan   = 5     # Number of channels to be concatenated (spikes) if Option = 1 or Option = 2

if Option == 0:
    
# %% 
    ''' {Approach 0. PCA over one channel of one cluster. The other clusters are projected 
                     on these axes.} '''

        # Create PC-Subspace and project clusters onto that subspace.
    pca_subspace = TF.PCA_dimred(n_pca, perc_pca, Suboption, Waveforms[0][:,0,:])
    Proj = []
    for i in range(len(Waveforms)):
        Proj.append(np.matmul(pca_subspace.components_,np.transpose(Waveforms[i][:,0,:])))
        
        # Plot Clusters if we have 2- or 3-D space
    if Suboption == 0:
        ax = TF.ClustPlot(n_pca, Proj)

elif Option == 1:
# %% 
    ''' {Approach 1. PCA over n channels with highest amplitude of one cluster. The other 
                     clusters are projected on these axes - Highest Amplitude for all of them}. '''
    
        # Extract n channels with highest amplitude from the averages
    Waveforms_HA_c, indices_maxn = TF.extract_HAwav(n_chan, Waveforms, av_Waveforms) 
    
        # Create PC-Subspace and project clusters onto that subspace.
    pca_subspace = TF.PCA_dimred(n_pca, perc_pca, Suboption, Waveforms_HA_c[0])
    Proj = [] 
    for i in range(len(Waveforms_HA_c)):
        Proj.append(np.matmul(pca_subspace.components_,np.transpose(Waveforms_HA_c[i])))
    
        # Plot Clusters if we have 2- or 3-D space
    if Suboption == 0:
        ax = TF.ClustPlot(n_pca, Proj)

elif Option == 2:
# %% 
    ''' {Approach 2. PCA over n channels with highest amplitude of one cluster. The other 
                    clusters are projected on these axes - SAME CHANNELS, not highest Amplitude}. '''
    
        # Extract n channels with highest amplitude from the averages and create PC-Subspace.
    Waveforms_HA_c, indices_maxn = TF.extract_HAwav_t(Option, n_chan, Waveforms, av_Waveforms, Clust_ref = 0) 
    pca_subspace = TF.PCA_dimred(n_pca, perc_pca, Suboption, Waveforms_HA_c[0])
        
        # Projection of all clusters in the PC-Space
    Proj = [] 
    for i in range(len(Waveforms_HA_c)):
        Proj.append(np.matmul(pca_subspace.components_,np.transpose(Waveforms_HA_c[i])))   
    
        # Plot Clusters if we have 2- or 3-D space
    if Suboption == 0:
        ax = TF.ClustPlot(n_pca, Proj)
        
elif Option == 3:               
# %% 
    ''' {Approach 3. PCA over ALL channels of all clusters. This approach is meant
         to be useful when a lot of channels are used so that information about all
         spikes is taken into account}. ''' 

        # Extract n = all channels with highest amplitude from the averages. 
    n_chan            = Waveforms[0].shape[1] # Number of Channels
    Waveforms_HA_c    = TF.extract_Allwav(Waveforms) # Concatenated Waves
    Waveforms_Allsamp = TF.conc_Allsamp(Waveforms_HA_c) # All Spike-waves
    
        # Create PC-Subspace and project clusters onto it.
    pca_subspace = TF.PCA_dimred(n_pca, perc_pca, Suboption, Waveforms_Allsamp)
    Proj = [] 
    for i in range(len(Waveforms_HA_c)):
        Proj.append(np.matmul(pca_subspace.components_,np.transpose(Waveforms_HA_c[i])))
     
        # Plot Clusters if we have 2- or 3-D space
    if Suboption == 0:
        ax = TF.ClustPlot(n_pca, Proj)
    
# %% Data from PCA and PC-Subspace
        
Variance_Ratio = pca_subspace.explained_variance_ratio_
S_Values       = pca_subspace.singular_values_
Tot_Var        = np.sum(Variance_Ratio)

# %% SPIKES PLOTS

if Print_S_Opt == 0:   # No Plot
    print('No spike waveforms plotted.')
    
elif Print_S_Opt == 1: # Plot a set of single spikes in different figures
    TF.PlotSingleWaveforms(Waveforms, av_Waveforms, Time, cl = 0)
    
elif Print_S_Opt == 2 or Print_S_Opt == 3: # Plot concatenated waveforms  
    Time = np.linspace(0,2*n_chan,Waveforms_HA_c[0].shape[1])
    Waveforms_HA_c_av = TF.extract_HAwav_av(Option, n_chan, Waveforms_HA_c) # Av. Waveforms
    if Option == 3: # No ref cluster, no indexes
        indices_maxn = []      
    TF.PlotWaveforms(Print_S_Opt, Option, Waveforms_HA_c, Waveforms_HA_c_av, Time, n_chan, indices_maxn)   
    
else: # Invalid option
    print('Please select a valid option for plotting spike waveforms.')
    
# %% Parameter to evaluate Classification - Mahalanobis Distance Metric
if Option_Mah == 0:
    print('No Mahalanobis Distance Metric Computed')
    
elif Option_Mah == 1:
    # Compute the Mahalanobis distance of every point to every distribution,
    # excluding the point from its own distribution to avoid biases (LOO Algorithm).
    
        # Compute Metric
    Av_dist_mat = TF.compute_distmat(Proj, metric = 'mahalanobis')
    
        # Plot Mahalanobis Distance Matrices
    TF.plotMahala_mat(Av_dist_mat, Option, Suboption, n_chan, Tot_Var)
    
# %%
    Av_dist_mat_sim = (Av_dist_mat + Av_dist_mat.transpose())/2
    Red_mat = pdist(Av_dist_mat_sim)   
    
    Link_ward   = TF.plot_hierarchical_dend(Red_mat, hier_method = 'ward')
    Link_single = TF.plot_hierarchical_dend(Red_mat, hier_method = 'single')
        
    np.save('Link_ward.npy',Link_ward)
    np.save('Link_single.npy',Link_single)
    
# =============================================================================
# # Create a structure of submatrices with pairs of cluster distances
#     
#     ii = 0
#     MD_Matrix = []
#     for i in range(len(clust_names)):
#         MD = []
#         jj = 0
#         for j in range(len(clust_names)):
#             MD.append(Av_dist_mat[ii:(ii+len(clust_names[i])), jj:(jj+len(clust_names[j]))])
#             jj = jj + len(clust_names[j]) 
#         
#         ii = ii + len(clust_names[i])
#         MD_Matrix.append(MD)
# else: 
#     print('Please select a valid option for computing Mahalanobis Distance Matrix')
#     
# # %% Find Final parameter
# Dist_min_cluster = []
# Ind_min_cluster  = []
# Metric_Opt = 0;
# for i in range(len(MD_Matrix)): # For each cluster
#     Dist_min_c = []
#     Ind_min_c  = []
#     for j in range(len(MD_Matrix[i])): # Each cluster
#         Dist_min = []
#         Ind_min  = []
#         for k in range(len(MD_Matrix[i][i])): # Each element in the diagonal of 1 cluster (Diag sub-matrix [i][i])        
#             if i != j: # Do not check with itself
#                 if Metric_Opt == 0:
#                     ind_m = np.argmin(MD_Matrix[i][j][k,:])
#                     # Metric = (MD_Matrix[i][j][k,ind_m] + MD_Matrix[j][i][ind_m,k])/2 # Consider both sym. values   
#                     Metric = MD_Matrix[i][j][k,ind_m]
#                 Dist_min.append(Metric)
#                 Ind_min.append(ind_m)     
#         Dist_min_c.append(Dist_min)
#         Ind_min_c.append(Ind_min)
#     Dist_min_cluster.append(Dist_min_c)
#     Ind_min_cluster.append(Ind_min_c)
# 
# 
# for i in range(len(Ind_min_cluster)): # For every cluster
#     for j in range(len(Ind_min_Cluster)): # For every cluster        
#         if i != j:
#             if len(Ind_min_cluster[i][j] >= Ind_min_cluster[j][i]):
#                 corr_neur.append(Ind_min_cluster[j][i])
#             else:
#                 corr_neur.append(Ind_min_cluster[i][j])
# =============================================================================
        
        
        
        
        