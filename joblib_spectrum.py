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

import compute_spectrum

net_heat = ContTempNetwork.load('fig3_growing_network300')
folder = '//scratch/tmp/180/skoove/growing_experiment300_temporal_heat/'

def worker(lamda):
    with open(folder + f'TvNS_selected/T{lamda:.11f}', 'rb') as f:
        dict_T = pickle.load(f)
    
    try:
        eigs = compute_spectrum.compute_spectrum(net=net_heat, list_T=dict_T['T'], lamda=lamda,
                                    force_csr=True,
                                    time_domain= list(np.arange(0, len(net_heat.times), 10))) #[len(net_heat.times)-2]) #list(np.arange(0, len(net_heat.times), 10)))
        spectrum={'lamda': f'{lamda:.11f}', 'spectrum': eigs}
        file= folder + f'spectrum_selected/spectrum{lamda:.11f}'
        with open(file, 'wb') as fopen:
            pickle.dump(spectrum, fopen)
    except ValueError:
        print('error with lamda')
    
    print(lamda)

lambdas = np.logspace(-3,1,10)

n_cpu = 50 #some appropriate number
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)