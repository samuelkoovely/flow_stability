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

import compute_S_rate

net_heat = ContTempNetwork.load('evolving_SBM_net3_b')
# net = ContTempNetwork.load('/home/b/skoove/Desktop/entropy/paper_data/socio_pat_primary_school/primaryschoolnet')

folder = '//scratch/tmp/180/skoove/evolving_SBM_heat/net3_b/'
#folder = '//scratch/tmp/180/skoove/primaryschoolnet_rw/'

# with open(folder + 'L', 'rb') as f:
#     dict_L = pickle.load(f)

def worker(lamda):
    with open(folder + f'T/T{lamda:.11f}', 'rb') as f:
        dict_T = pickle.load(f)

    
    try:
        # S = compute_S_rate.compute_entropy_rate(net=net_heat, list_T=dict_T['T'],
        #                             lamda=lamda, force_csr=True,
        #                             time_domain= list(np.arange(0, len(net_heat.times)-2))) #[len(net_heat.times)-2]) #list(np.arange(0, len(net_heat.times), 10)))
        S = compute_S_rate.compute_conditional_entropy(net=net_heat, list_T=dict_T['T'],
                                    lamda=lamda, force_csr=True,
                                    time_domain= list(np.arange(0, len(net_heat.times)-2))) #[len(net_heat.times)-2]) #list(np.arange(0, len(net_heat.times), 10)))
        
        # S = compute_S_rate.compute_instantaneous_entropy_rate(net=net_heat, list_T=dict_T['T'], list_L=dict_L['L'],
        #                             lamda=lamda, force_csr=True,
        #                             time_domain= list(np.arange(0, len(net_heat.times)-2))) #[len(net_heat.times)-2]) #list(np.arange(0, len(net_heat.times), 10)))
        
        S_rate={'lamda': f'{lamda:.11f}', 'S_rate': S}
        file= folder + f'S_rate/S_rate{lamda:.11f}'
        with open(file, 'wb') as fopen:
            pickle.dump(S_rate, fopen)
    except ValueError:
        S_rate={'lamda': f'{lamda:.11f}', 'S_rate': 100}
        print('error with lamda')
    
    print(lamda)

lambdas = np.logspace(-5,0,10)

n_cpu = 10 #some appropriate number
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)