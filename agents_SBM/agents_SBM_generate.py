import numpy as np
from SynthTempNetwork import Individual, SynthTempNetwork
from TemporalNetwork import ContTempNetwork

def make_step_block_probs(deltat1, num_cores=27, basis_num_communities = 3, powers_num_communities = [3, 2, 1], list_p_within_community = [49/50] * len([27, 9, 3])):
    """
    Returns a function that generates the block probability matrix as a function of time.

    Parameters:
    - deltat1 (int): Length of each temporal step.
    - num_cores (int): Total number of cores in the network.
    - basis_num_communities (int): The base number of communities for symmetry.
    - powers_num_communities (list of int): Powers of the base defining the number of communities in each stage.
    - list_p_within_community (list of float): List of within-community interaction probabilities for each stage.

    """
    
    list_num_communities=list(np.power(basis_num_communities, powers_num_communities))
    
    def calculate_pout(p_within, community_size, total_size):
        k = (1 / p_within - community_size) / (total_size - community_size)
        return k * p_within

    def generate_block_matrix(num_communities, p_within_base):
        # Determine community sizes
        community_size = num_cores // num_communities
        p_within = p_within_base / community_size  # Adjust within probability by the number of cores per community
        pout = calculate_pout(p_within, community_size, num_cores)

        # Create block matrix
        block_matrix = np.zeros((num_cores, num_cores))
        for i in range(num_communities):
            start = i * community_size
            end = start + community_size
            block_matrix[start:end, start:end] = p_within

        block_matrix[block_matrix == 0] = pout
        return block_matrix

    # Precompute the matrices for different stages
    stage_matrices = []
    for num_communities, p_within_community in zip(list_num_communities, list_p_within_community):
        stage_matrices.append(generate_block_matrix(num_communities, p_within_community))

    def block_mod_func(t):
        total_stages = len(stage_matrices)
        stage_duration = deltat1

        for stage_index in range(total_stages):
            if t >= stage_index * stage_duration and t < (stage_index + 1) * stage_duration:
                return stage_matrices[stage_index]

        print("Warning: t is out of bounds. Returning identity matrix.")
        return np.eye(num_cores)

    return block_mod_func


def generate_agents_SBM(block_mod_func, inter_tau = 2, activ_tau = 2, n_per_group = 3, n_groups = 27, t_start = 0, t_end = 300):

    # create agents for the simlation
    individuals = []

    for g in range(n_groups):

        individuals.extend([Individual(i, inter_distro_scale=inter_tau,
                                        activ_distro_scale=activ_tau,
                                        group=g) for i in range(g*n_per_group,(g+1)*n_per_group)])


    # run simulation
    sim = SynthTempNetwork(individuals=individuals, t_start=t_start, t_end=t_end,
                        next_event_method='block_probs_mod',
                        block_prob_mod_func=block_mod_func)

    print('running simulation')
    sim.run(save_all_states=True, save_dt_states=True, verbose=False)


    temporal_net = ContTempNetwork(source_nodes=sim.indiv_sources,
                            target_nodes=sim.indiv_targets,
                            starting_times=sim.start_times,
                            ending_times=sim.end_times,
                            merge_overlapping_events=True)
    
    return temporal_net