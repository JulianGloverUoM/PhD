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
import NF4
import scipy as sp
import scipy.stats as scp


#############################################################################
#############################################################################


N = 80

L = 2
# seeds = [0, 7, 16, 17, 21, 23, 25, 26, 30]

seeds_2 = [10, 12, 17, 27, 33, 35, 39, 40, 41, 45, 49]

data_list = []

stresses_list = []
edge_lengths = []

for seed in range(200):
    # if seed in seeds:
    #     continue
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
    ) = NF4.restructure_PBC_data(pbc_edges, pbc_nodes, pbc_adjacency_matrix, L)

    edges = adjacency_matrix.dot(nodes)

    for edge in edges:
        edge_lengths.append(np.linalg.norm(edge))


sample_lambda = 1 / (sum(edge_lengths) / len(edge_lengths))
x_vals = np.linspace(min(edge_lengths), max(edge_lengths), 100)
exp_dist = sample_lambda * np.exp(-sample_lambda * x_vals)
exp_cdf = 1 - exp_dist / sample_lambda

loc, scale = scp.expon.fit(edge_lengths)  # Fit the left part of the distrib.
Test = scp.kstest(edge_lengths, "expon", args=(loc, scale))  # KS Test
figure_0 = plt.figure(0)
plt.plot(x_vals, exp_dist, color="r", linewidth=0.5)
plt.hist(edge_lengths, 100, density=True, color="b")
plt.title("Histogram of edge lengths for random seed = {}".format(seed))
plt.savefig("Edge_lengths_PDF_{}".format(seed))
plt.show()

figure_1 = plt.figure(1)
plt.hist(edge_lengths, 100, density=True, histtype="step", cumulative=True, color="b")
plt.plot(x_vals, exp_cdf, color="r", linewidth=0.5)
plt.title("Cumulative histogram of edge lengths for random seed = {}".format(seed))
plt.savefig("Edge_lengths_CDF_{}".format(seed))
plt.show()
print(
    "D_pm, p_value) = ",
    (
        Test[0],
        Test[1],
    ),
)
#     ,
#     " with sample lambda = ",
#     sample_lambda,
# )
# fig = plt.figure()
# PBC_network.PlotNetwork_2(nodes, adjacency_matrix, 1, 0)
# plt.title(str("random seed = {}".format(i)))

# (data, stresses) = NF4.Single_realisation(
#     L,
#     nodes,
#     adjacency_matrix,
#     boundary_nodes,
#     0.9,
#     0.003,
#     11,
#     Plot_stress_results=True,
#     Plot_networks=False,
# )
# data_list.append(data)
# stresses_list.append(stresses)

# sample_lambda = 1 / (sum(edge_lengths) / len(edge_lengths))
# x_vals = np.linspace(min(edge_lengths), max(edge_lengths), 100)
# exp_dist = sample_lambda * np.exp(-sample_lambda * x_vals)
# exp_cdf = 1 - exp_dist / sample_lambda

# loc, scale = scp.expon.fit(edge_lengths, floc=0)  # Fit the left part of the distrib.
# Test = scp.kstest(edge_lengths, "expon", args=(loc, scale))  # KS Test
# figure_0 = plt.figure(0)
# plt.plot(x_vals, exp_dist, color="r")
# plt.hist(edge_lengths, 200, density=True, color="b")
# plt.title("Histogram of edge lengths for random seed = {}".format(seed))
# plt.savefig("Edge_lengths_PDF_{}".format(seed))
# plt.show()

# figure_1 = plt.figure(1)
# plt.hist(edge_lengths, 200, density=True, histtype="step", cumulative=True, color="b")
# plt.plot(x_vals, exp_cdf, color="r")
# plt.title("Cumulative histogram of edge lengths for random seed = {}".format(seed))
# plt.savefig("Edge_lengths_CDF_{}".format(seed))
# plt.show()
# print(
#     "For seed {}, (D_pm, p_value) = ".format(seed),
#     (
#         Test[0],
#         Test[1],
#     ),
# )
