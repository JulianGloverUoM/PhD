######################################
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import time
import os
import sys

file_path = os.path.realpath(__file__)
sys.path.append(file_path)
import PBC_network  # noqa
import NF_RK23
import scipy as sp
import scipy.stats as scp


#############################################################################
#############################################################################


L = 1

N = 15

norms = []

ratios = []

nodes_initial = []

nodes_deform = []

nodes_inverted = []

for seed in range(10):
    Network = PBC_network.CreateNetwork(N, L, seed)
    pbc_edges = Network[0][2]
    pbc_nodes = Network[1][2]
    pbc_adjacency_matrix = Network[2][1]

    (
        nodes,
        edges,
        adjacency_matrix,
        boundary_nodes,
        top_nodes,
        bot_nodes,
        left_nodes,
        right_nodes,
    ) = NF_RK23.restructure_PBC_data(pbc_edges, pbc_nodes, pbc_adjacency_matrix, L)

    y_initial_relaxed = NF_RK23.RK23_timestepper(
        L,
        nodes,
        adjacency_matrix,
        boundary_nodes,
        0.8 * NF_RK23.vector_of_magnitudes(adjacency_matrix.dot(nodes)),
        0,
        Plot_networks=True,
    )[3]

    y_deform_relaxed = NF_RK23.RK23_timestepper(
        L,
        y_initial_relaxed,
        adjacency_matrix,
        boundary_nodes,
        0.8 * NF_RK23.vector_of_magnitudes(adjacency_matrix.dot(nodes)),
        0.01,
        Plot_networks=True,
    )[3]

    input_y_deform_relaxed = NF_RK23.shear_x(y_deform_relaxed, -0.01)

    y_relaxed = NF_RK23.RK23_timestepper(
        L,
        input_y_deform_relaxed,
        adjacency_matrix,
        boundary_nodes,
        0.8 * NF_RK23.vector_of_magnitudes(adjacency_matrix.dot(nodes)),
        0,
        Plot_networks=True,
    )[3]

    print(np.linalg.norm(y_initial_relaxed - y_deform_relaxed))
    print(np.linalg.norm(y_initial_relaxed - y_relaxed))
    print(
        "Ratio = ",
        np.linalg.norm(y_initial_relaxed - y_relaxed)
        / np.linalg.norm(y_initial_relaxed - y_deform_relaxed),
    )

    norms.append(
        [
            np.linalg.norm(y_initial_relaxed - y_deform_relaxed),
            np.linalg.norm(y_initial_relaxed - y_relaxed),
        ]
    )

    ratios.append(
        np.linalg.norm(y_initial_relaxed - y_relaxed)
        / np.linalg.norm(y_initial_relaxed - y_deform_relaxed)
    )

    nodes_initial.append(y_initial_relaxed)
    nodes_deform.append(y_deform_relaxed)
    nodes_inverted.append(y_relaxed)
