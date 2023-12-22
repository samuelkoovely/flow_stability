#!/bin/bash

# lamdas
LAMBDAS=(0.0001 0.000207 0.000428 0.000886 0.001833 0.003793 0.007848 0.016238 0.033598 0.069519 0.143845 0.297635 0.615848 1.274275 2.636651 5.455595 11.288379 23.357215 48.329302 100.0)

for l in ${LAMBDAS[@]}
do
    echo $l 
    screen -d -m -S $l /Users/samuelkoovely/Documents/GitHub/flow_stability/flow_stability_env/bin/python3 entropy_experiment.py --lamda $l
done