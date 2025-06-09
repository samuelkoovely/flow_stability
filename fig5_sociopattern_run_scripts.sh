#!/bin/bash

#
# Flow stability for dynamic community detection https://arxiv.org/abs/2101.06131v2
#
# Copyright (C) 2021 Alexandre Bovet <alexandre.bovet@maths.ox.ac.uk>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


# scripts to perform the clustering of the primary school dataset


# numpy/scipy should use 1 core since we have a multiprocessing script
export OPENBLAS_NUM_THREADS=1

NCPU=8

JNAME="sociopat_pm"

NETFILENAME="primaryschoolnet.pickle"

RUNDIR="$HOME/rundir"

DATADIRBASE="$HOME/datadir"

mkdir $DATADIRBASE

LAPTRANSDIR="$DATADIRBASE/lapl_transmat"

LAPINTERTRANSDIR="$DATADIRBASE/lapl_intertransmat5"

TRANSMATDIR="$DATADIRBASE/transmatgrid"

INTEGDIR="$DATADIRBASE/integralgrid"


CLUSTSDIRT="$DATADIRBASE/clustersT"

CLUSTNUMREPEAT=50

NETNAME="primaryschool"

# laplacians and transition matrices resolution grid
SLICELENGTH="$((60*15))" # slice length = 15 min


# final grid resolution in unit of slices
INT_LENGTH="$((2))" # 30min

mkdir $LAPTRANSDIR
mkdir $LAPINTERTRANSDIR
mkdir $TRANSMATDIR
mkdir $CLUSTSDIRI
mkdir $CLUSTSDIRT
mkdir $INTEGDIR


# waiting times = 20s 20min 3h 1day
TAUWS=(20 "$((60*20))" "$((60*60))" "$((60*60*3))" "$((60*60*24))" 2 4.3 9.3 20 200 2000 20000 63 630 6300 63000 36 112 360 1120 3600 11200 36000 112000)


# compute laplacians and intertransition matrices
python -u $RUNDIR/run_laplacians_transmat.py \
        --datadir $DATADIRBASE \
        --savedir $LAPINTERTRANSDIR \
        --net_filename $NETFILENAME \
        --net_name $NETNAME \
        --not_expm_transmat \
        --slice_length $SLICELENGTH \
        --ncpu $NCPU \
        --tau_w_list ${TAUWS[*]} \
        --save_slice_trans \
        --save_inter_trans \
        --not_expm_transmat \
        > output_${JNAME}_lptm.txt 2> error_${JNAME}_lptm.txt
#         
        
# compute autocov_grid        
python -u $RUNDIR/make_autocov_integral.py \
        --datadir $LAPINTERTRANSDIR \
        --savedir $INTEGDIR \
        --net_name $NETNAME \
        --int_length $INT_LENGTH \
        --ncpu $NCPU \
        --only_lin_transmats \
        > output_${JNAME}_intg.txt 2> error_${JNAME}_intg.txt

# compute clustering of integral
# 
python -u $RUNDIR/run_clustering.py \
        --datadir $INTEGDIR \
        --savedir $CLUSTSDIRI \
        --ncpu $NCPU \
        --num_repeat $CLUSTNUMREPEAT \
        --net_name $NETNAME \
        > output_${JNAME}_clust.txt 2> error_${JNAME}_clust.txt

