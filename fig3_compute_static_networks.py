from joblib import Parallel, delayed
import numpy as np
from SynthTempNetwork import Individual, SynthTempNetwork
from TemporalNetwork import ContTempNetwork, StaticTempNetwork
from FlowStability import FlowIntegralClustering
import pickle

import matplotlib.pyplot as plt
import matplotlib

from scipy.sparse import (lil_matrix, dok_matrix, diags, eye, isspmatrix_csr, isspmatrix,
                          csr_matrix, coo_matrix, csc_matrix)



def worker(i):
    #weighted
    #Combinatorial Laplacian
    static_net_weighted_heat = StaticTempNetwork(times = net.times, adjacency = static_net_adj)
    static_net_weighted_heat.compute_laplacian_matrices(random_walk = False)

    static_net_weighted_heat.compute_inter_transition_matrices(lamda=i, dense_expm=False, use_sparse_stoch=False)
    static_net_weighted_heat.compute_transition_matrices(lamda=i)

    # #Random Walk Laplacian
    # static_net_weighted = StaticTempNetwork(times = net.times, adjacency = static_net_adj)
    # static_net_weighted.compute_laplacian_matrices(random_walk = True)

    # static_net_weighted.compute_inter_transition_matrices(lamda=i, dense_expm=False, use_sparse_stoch=False)
    # static_net_weighted.compute_transition_matrices(lamda=i)

    # #unweighted

    # #Combinatorial Laplacian
    # static_net_heat = StaticTempNetwork(times = net.times, adjacency = static_net_adj_unweighted)
    # static_net_heat.compute_laplacian_matrices(random_walk = False)

    # static_net_heat.compute_inter_transition_matrices(lamda=i, dense_expm=False, use_sparse_stoch=False)
    # static_net_heat.compute_transition_matrices(lamda=i)

    # #Random Walk Laplacian
    # static_net = StaticTempNetwork(times = net.times, adjacency = static_net_adj_unweighted)
    # static_net.compute_laplacian_matrices(random_walk = True)

    # static_net.compute_inter_transition_matrices(lamda=i, dense_expm=False, use_sparse_stoch=False)
    # static_net.compute_transition_matrices(lamda=i)

    
    # Saving Compuations
    file1=f'//scratch/tmp/180/skoove/growing_experiment300_static_weighted_heat/inter_Tplot/inter_T{i:.11f}'
    file2=f'//scratch/tmp/180/skoove/growing_experiment300_static_weighted_heat/Tplot/T{i:.11f}'
    # file3=f'//scratch/tmp/180/skoove/experiment_static_weighted/inter_T/inter_T{i:.06f}'
    # file4=f'//scratch/tmp/180/skoove/experiment_static_weighted/T/T{i:.06f}'
    # file5=f'//scratch/tmp/180/skoove/experiment_short_static_heat/test_interT{i:.06f}'
    # file6=f'//scratch/tmp/180/skoove/experiment_short_static_heat/test_T{i:.06f}'
    # file7=f'//scratch/tmp/180/skoove/experiment_short_static/test_interT{i:.06f}'
    # file8=f'//scratch/tmp/180/skoove/experiment_short_static/test_T{i:.06f}'


    weighted_heat_inter_T={'lamda': i, 'inter_T': static_net_weighted_heat.inter_T[i]}
    weighted_heat_T={'lamda': i, 'T': static_net_weighted_heat.T[i]}
    with open(file1, 'wb') as fopen:
        pickle.dump(weighted_heat_inter_T, fopen)
    with open(file2, 'wb') as fopen:
        pickle.dump(weighted_heat_T, fopen)
    
    # weighted_inter_T={'lamda': i, 'inter_T': static_net_weighted.inter_T[i]}
    # weighted_T={'lamda': i, 'T': static_net_weighted.T[i]}
    # with open(file3, 'wb') as fopen:
    #     pickle.dump(weighted_inter_T, fopen)
    # with open(file4, 'wb') as fopen:
    #     pickle.dump(weighted_T, fopen)
    print(i)



lambdas = np.logspace(-2.5,0,6)

net = ContTempNetwork.load('fig3_network300_temporal_heat')

#Generating Networks

static_net_adj = net.compute_static_adjacency_matrix()

#entries are sum of durations of events, need to set to 1
static_net_adj_unweighted = static_net_adj.toarray()
static_net_adj_unweighted[static_net_adj_unweighted != 0] = 1
static_net_adj_unweighted = csr_matrix(static_net_adj_unweighted)

n_cpu = 50 #some appropriate number


Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)