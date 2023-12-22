import numpy as np
from SynthTempNetwork import Individual, SynthTempNetwork
from TemporalNetwork import ContTempNetwork, StaticTempNetwork
from FlowStability import FlowIntegralClustering
import pickle

import matplotlib.pyplot as plt
import matplotlib

from scipy.sparse import (lil_matrix, dok_matrix, diags, eye, isspmatrix_csr, isspmatrix,
                          csr_matrix, coo_matrix, csc_matrix)

import argparse

ap = argparse.ArgumentParser()

ap.add_argument('--lamda', required=True, type=float)

inargs = vars(ap.parse_args())
lamda = inargs['lamda']

print(lamda)

static_net_weighted_heat = StaticTempNetwork.load('fig3_network_static_weighted_heat')

file=f'test_lamda{lamda:.06f}'

print(file)

static_net_weighted_heat.compute_inter_transition_matrices(lamda=lamda, dense_expm=False, use_sparse_stoch=False)
static_net_weighted_heat.compute_transition_matrices(lamda=lamda)
static_net_weighted_heat.compute_vonNeumann_entropy(lamda=lamda, force_csr=True, time_domain = static_net_weighted_heat.times[-1:])

save_dict={'lamda': lamda, 'inter_T': static_net_weighted_heat.inter_T[lamda], 'T': static_net_weighted_heat.T[lamda],
           'vNS': static_net_weighted_heat.vNS[lamda]}
with open(file, 'wb') as fopen:
    pickle.dump(save_dict, fopen)