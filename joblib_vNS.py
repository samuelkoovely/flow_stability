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

import compute_vNS

net_heat = ContTempNetwork.load('/home/b/skoove/Desktop/entropy/paper_data/socio_pat_primary_school/primaryschoolnet')
folder = '//scratch/tmp/180/skoove/primaryschoolnet_rw/'

def worker(lamda):
    with open(folder + f'Tselected/T{lamda:.11f}', 'rb') as f:
        dict_T = pickle.load(f)
    
    try:
        S = compute_vNS.compute_vonNeumann_entropy(net=net_heat, list_T=dict_T['T'], lamda=lamda,
                                    force_csr=True,
                                    time_domain= list(np.arange(0, len(net_heat.times)-2))) #[len(net_heat.times)-2]) #list(np.arange(0, len(net_heat.times), 10)))
        vNS={'lamda': f'{lamda:.11f}', 'vNS': S}
        file= folder + f'vNSselected_hr/vNS{lamda:.11f}'
        with open(file, 'wb') as fopen:
            pickle.dump(vNS, fopen)
    except ValueError:
        vNS={'lamda': f'{lamda:.11f}', 'vNS': 2}
        print('error with lamda')
    
    print(lamda)

lambdas = np.logspace(-5,0,10)

n_cpu = 60 #some appropriate number
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)