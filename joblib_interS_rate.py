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

import compute_interS_rate

net_heat = ContTempNetwork.load('fig3_growing_network300_temporal_heat')


def worker(lamda):
    with open(f'//scratch/tmp/180/skoove/growing_experiment300_temporal_heat/inter_TvNS_selected/inter_T{lamda:.11f}', 'rb') as f:
        dict_inter_T = pickle.load(f)
    
    try:
        inter_S = compute_interS_rate.compute_inter_entropy_rate(net=net_heat, list_inter_T=dict_inter_T['inter_T'], lamda=lamda,
                                    force_csr=True,
                                    time_domain= list(np.arange(0, len(net_heat.times), 10))) #[len(net_heat.times)-2]) #list(np.arange(0, len(net_heat.times), 10)))
        inter_S_rate={'lamda': f'{lamda:.11f}', 'inter_S_rate': inter_S}
        file=f'//scratch/tmp/180/skoove/growing_experiment300_temporal_heat/inter_S_rate_selected/inter_S_rate{lamda:.11f}'
        with open(file, 'wb') as fopen:
            pickle.dump(inter_S_rate, fopen)
    except ValueError:
        S_rate={'lamda': f'{lamda:.11f}', 'S_rate': 2}
        print('error with lamda')
    
    print(lamda)

lambdas = np.logspace(-3,1,10) # np.logspace(-5,0,200)

n_cpu = 50 #some appropriate number
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)