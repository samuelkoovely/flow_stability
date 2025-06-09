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
from scipy.linalg import logm as d_logm

import compute_vNS

net = ContTempNetwork.load('fig3_network250_temporal_heat')


def worker(lamda):
    with open(f'//scratch/tmp/180/skoove/experiment250_temporal_static_weighted_heat/sam_slow/sam{lamda:.11f}', 'rb') as f:
        sam = pickle.load(f)
    

    sam = sam.toarray()
    rho = sam/np.trace(sam)
    rhologrho = rho @ d_logm(rho)
    vNS = - 1/np.log(net.num_nodes) * np.trace(rhologrho)
    del sam
    del rho
    del rhologrho
    
    file=f'//scratch/tmp/180/skoove/experiment250_temporal_static_weighted_heat/vNSsam_slow/vNS{lamda:.11f}'
    with open(file, 'wb') as fopen:
        pickle.dump(vNS, fopen)

    print(lamda)

lambdas = np.logspace(-10,-6,400)

n_cpu = 50 #some appropriate number
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)