# Code will be given a p, lambda_1, lambda_2 min/max. The it will calculate all the quantities listed below
# Fibre stretches wrt rest (initial) state
# Fibre stretches wrt initial equilibrium (reference) state
# Initial and current fibre orientations
# Predictions of fibre stretches wrt reference state using initial and deformed theta
# Chi wrt reference state
# produce 6 plots and save them as .png's

######################################
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import copy
import time
import os
import sys
import pickle

file_path = os.path.realpath(__file__)
sys.path.append(file_path)
import Dilation_radau  # noqa
import PBC_network  # noqa

######################################

#############################################################################
#############################################################################

# Applies homogenous stretch/dilation deformation to Nx2 array of N nodes


def dilation_deformation(input_nodes, lambda_1, lambda_2):
    return np.array([np.array([[lambda_1, 0], [0, lambda_2]]).dot(item) for item in input_nodes])


def invert_dilation(input_nodes, lambda_1, lambda_2):
    return np.array(
        [np.array([[1 / lambda_1, 0], [0, 1 / lambda_2]]).dot(item) for item in input_nodes]
    )


#############################################################################
#############################################################################

# Normalises the rows of an array


def normalise_elements(input_array):
    return np.array(
        [(np.array(vector) / np.sqrt(np.einsum("i,i", vector, vector))) for vector in input_array]
    )


# Takes in a array and outputs the magnitudes of the rows in the array


def vector_of_magnitudes(input_array):
    return np.array([np.sqrt(np.einsum("i,i", vector, vector)) for vector in input_array])


# Computes the forbenius norm of a matrix, more efficient than inbuilt np.linalg.norm function


def frobenius_norm(input_array):
    return np.sqrt(np.einsum("ij,ij", input_array, input_array))


#############################################################################
#############################################################################


def restructure_PBC_data(pbc_edges, pbc_nodes, pbc_incidence_matrix, L):
    PBC_data = []

    for edge in pbc_edges:
        if len(edge) == 2:
            PBC_data.append(
                [
                    pbc_edges.index(edge),
                    np.nonzero(pbc_incidence_matrix[pbc_edges.index(edge)])[0],
                    edge[0][1],
                    edge[1][0],
                ]
            )
        if len(edge) == 3:
            PBC_data.append(
                [
                    pbc_edges.index(edge),
                    np.nonzero(pbc_incidence_matrix[pbc_edges.index(edge)])[0],
                    edge[0][1],
                    edge[1],
                    edge[2][0],
                ]
            )

    incidence_matrix = np.copy(pbc_incidence_matrix)

    edges = copy.deepcopy(pbc_edges)
    nodes = np.copy(pbc_nodes)

    incidence_matrix = np.block(
        [
            [incidence_matrix, np.zeros([len(pbc_edges), 2 * len(PBC_data)])],
            [np.zeros([len(PBC_data), 2 * len(PBC_data) + np.shape(incidence_matrix)[1]])],
        ]
    )

    if L == 1:
        d = 0.5
    else:
        d = 1

    for item in PBC_data:
        edge_index = item[0]
        node_indices = item[1]
        # This makes sure the nodes (a,b) in node_indices are ordered such that
        # b is connected to the last entry of the item.
        if np.linalg.norm(nodes[node_indices[0]] - np.array(item[-1])) < d:
            node_indices = np.flip(node_indices, 0)
        if len(item) == 4:
            # Truncating old boundary edges and adding 'new' edges by
            # adding the end segment as a new edge, and remove that segement (and midde part)
            edges.append([edges[edge_index][-1]])
            edges[edge_index].remove(edges[edge_index][-1])

            # Removing the node connectivity across the boundary from the adjacency matrix
            # and replacing it with connections to 'new' nodes added on the boundary
            incidence_matrix[edge_index] = 0
            incidence_matrix[edge_index][node_indices[0]] = 1
            incidence_matrix[edge_index][len(nodes)] = -1
            nodes = np.append(nodes, np.array([item[2]]), axis=0)

            incidence_matrix[len(edges) - 1] = 0
            incidence_matrix[len(edges) - 1][node_indices[1]] = 1
            incidence_matrix[len(edges) - 1][len(nodes)] = -1
            nodes = np.append(nodes, np.array([item[3]]), axis=0)
        else:
            edges.append([edges[edge_index][-1]])
            edges[edge_index].remove(edges[edge_index][-1])
            edges[edge_index].remove(edges[edge_index][-1])

            incidence_matrix[edge_index] = 0
            incidence_matrix[edge_index][node_indices[0]] = 1
            incidence_matrix[edge_index][len(nodes)] = -1
            nodes = np.append(nodes, np.array([item[2]]), axis=0)

            incidence_matrix[len(edges) - 1] = 0
            incidence_matrix[len(edges) - 1][node_indices[1]] = 1
            incidence_matrix[len(edges) - 1][len(nodes)] = -1
            nodes = np.append(nodes, np.array([item[-1]]), axis=0)

    # All nodes from nodes[boundary_nodes] inclusive are on the boundary and
    # thus fixed after deformation

    boundary_nodes = len(pbc_nodes)

    top_nodes = []

    bot_nodes = []

    left_nodes = []

    right_nodes = []

    for i in range(len(nodes[boundary_nodes:])):
        if nodes[i + boundary_nodes][1] == L:
            top_nodes.append(i + boundary_nodes)
        if nodes[i + boundary_nodes][1] == 0:
            bot_nodes.append(i + boundary_nodes)
        if nodes[i + boundary_nodes][0] == 0:
            left_nodes.append(i + boundary_nodes)
        if nodes[i + boundary_nodes][0] == L:
            right_nodes.append(i + boundary_nodes)

    return (
        nodes,
        edges,
        incidence_matrix,
        boundary_nodes,
        top_nodes,
        bot_nodes,
        left_nodes,
        right_nodes,
    )


#############################################################################
#############################################################################


def produce_data(L, density, p, seed, lambda_1, lambda_2):
    Network = PBC_network.CreateNetwork(int(5.637 * density) * L**2, L, seed)

    pbc_edges = Network[0][2]
    pbc_nodes = Network[1][2]
    pbc_incidence_matrix = Network[2][1]

    (
        nodes,
        edges,
        incidence_matrix,
        boundary_nodes,
        top_nodes,
        bot_nodes,
        left_nodes,
        right_nodes,
    ) = Dilation_radau.restructure_PBC_data(pbc_edges, pbc_nodes, pbc_incidence_matrix, L)

    # print("Initial fiber lengths multiplier = ", fibre_lengths_multiplier)

    initial_lengths = p * vector_of_magnitudes(incidence_matrix.dot(nodes))
    if p != 1:
        prestress_data = Dilation_radau.Radau_timestepper_dilation(
            L, nodes, incidence_matrix, boundary_nodes, initial_lengths, 1, 1, False
        )
        nodes = prestress_data[-2]

    data = []
    data.append(
        Dilation_radau.Radau_timestepper_dilation(
            L,
            nodes,
            incidence_matrix,
            boundary_nodes,
            initial_lengths,
            lambda_1[0],
            lambda_2[0],
            False,
        )
    )
    data.append(
        Dilation_radau.Radau_timestepper_dilation(
            L,
            nodes,
            incidence_matrix,
            boundary_nodes,
            initial_lengths,
            lambda_1[-1],
            lambda_2[-1],
            False,
        )
    )

    reference_lengths = vector_of_magnitudes(incidence_matrix.dot(nodes))
    initial_theta = Dilation_radau.Orientation_distribution(nodes, incidence_matrix, False)

    current_thetas = [
        Dilation_radau.Orientation_distribution(item[-2], incidence_matrix, False) for item in data
    ]
    current_lengths = [vector_of_magnitudes(incidence_matrix.dot(item[-2])) for item in data]

    reference_stretches = [current_lengths[i] / reference_lengths for i in range(2)]

    chi = [
        Dilation_radau.chi_undeformed(
            reference_stretches[i], initial_theta, lambda_1[i], lambda_2[i]
        )
        for i in range(2)
    ]

    chi_def = [
        Dilation_radau.chi_deformed(reference_stretches[i], initial_theta, lambda_1[i], lambda_2[i])
        for i in range(2)
    ]

    predicted_stretches = []
    for i in range(2):
        predicted_stretches.append(
            [
                Dilation_radau.affine_stretch_calc(item, lambda_1[i], lambda_2[i])
                for index, item in enumerate(initial_theta)
            ]
        )

    predicted_stretches_deformed = []
    for i in range(2):
        predicted_stretches_deformed.append(
            [
                Dilation_radau.affine_stretch_def_calc(item, lambda_1[i], lambda_2[i])
                for index, item in enumerate(current_thetas[i])
            ]
        )

    predicted_orientations = [
        Dilation_radau.affine_theta_calc(lambda_1[i], lambda_2[i], initial_theta) for i in range(2)
    ]

    return (
        p,
        density,
        seed,
        lambda_1,
        lambda_2,
        chi,
        chi_def,
        reference_stretches,
        predicted_stretches,
        predicted_stretches_deformed,
        initial_theta,
        current_thetas,
        predicted_orientations,
        data,
    )


def produce_and_save_plots(
    p,
    density,
    seed,
    lambda_1,
    lambda_2,
    chi,
    chi_def,
    reference_stretches,
    predicted_stretches,
    predicted_stretches_deformed,
    initial_theta,
    current_thetas,
    predicted_orientations,
):
    fig_min, axs_min = plt.subplots(2, 3, figsize=(16, 8))  # , layout="constrained")
    fig_max, axs_max = plt.subplots(2, 3, figsize=(16, 8))  # , layout="constrained")

    # Hist stretch
    i = 0

    axs_min[0, 2].hist(reference_stretches[i], bins=100, density=True)
    axs_min[0, 2].hist(predicted_stretches[i], bins=100, alpha=0.8, color="orange", density=True)
    axs_min[0, 2].set_xlabel(r"$\lambda_a$", fontsize=13)
    axs_min[0, 2].set_ylabel(r"$p(\lambda_a)$")
    axs_min[0, 2].legend(["Numerical results", "Affine prediction"])
    # plt.title("$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[0], lambda_2[0]))
    # plt.savefig("lambda_hist_min_p{}.png".format(p))

    i = -1

    axs_max[0, 2].hist(reference_stretches[i], bins=100, density=True)
    axs_max[0, 2].hist(predicted_stretches[i], bins=100, alpha=0.8, color="orange", density=True)
    axs_max[0, 2].set_xlabel(r"$\lambda_a$", fontsize=13)
    axs_max[0, 2].set_ylabel(r"$p(\lambda_a)$")
    axs_max[0, 2].legend(["Numerical results", "Affine prediction"])
    # plt.title("$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[1], lambda_2[1]))
    # plt.savefig("lambda_hist_max_p{}.png".format(p))

    # Hist theta
    i = 0

    axs_min[1, 2].hist(current_thetas[i], bins=100, density=True)
    axs_min[1, 2].hist(predicted_orientations[i], bins=100, alpha=0.8, color="orange", density=True)
    axs_min[1, 2].set_xlabel(r"$\theta$", fontsize=13)
    axs_min[1, 2].set_ylabel(r"$p(\theta)$")
    axs_min[1, 2].legend(["Numerical results", "Affine prediction"])
    # plt.title("$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[0], lambda_2[0]))
    # plt.savefig("theta_hist_min_p{}.png".format(p))

    i = -1

    axs_max[1, 2].hist(current_thetas[i], bins=100, density=True)
    axs_max[1, 2].hist(predicted_orientations[i], bins=100, alpha=0.8, color="orange", density=True)
    axs_max[1, 2].set_xlabel(r"$\theta$", fontsize=13)
    axs_max[1, 2].set_ylabel(r"$p(\theta)$")
    axs_max[1, 2].legend(["Numerical results", "Affine prediction"])
    # plt.title("$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[1], lambda_2[1]))
    # plt.savefig("theta_hist_max_p{}.png".format(p))

    # Pdf stretch

    i = 0

    axs_min[0, 0].hist2d(reference_stretches[i], initial_theta, bins=100, density=True)
    axs_min[0, 0].plot(predicted_stretches[0], initial_theta, "ko", markersize=1, alpha=1)
    axs_min[0, 0].set_xlabel(r"$\lambda_a$", fontsize=13)
    axs_min[0, 0].set_ylabel(r"$\Theta$")
    axs_min[0, 0].legend(["Numerical results", "Affine prediction"])
    # plt.title("$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[0], lambda_2[0]))
    # plt.savefig("lambda_pdf_min_p{}.png".format(p))

    i = -1

    axs_max[0, 0].hist2d(reference_stretches[i], initial_theta, bins=100, density=True)
    axs_max[0, 0].plot(predicted_stretches[i], initial_theta, "ko", markersize=1, alpha=1)
    axs_max[0, 0].set_xlabel(r"$\lambda_a$", fontsize=13)
    axs_max[0, 0].set_ylabel(r"$\Theta$")
    axs_max[0, 0].legend(["Numerical results", "Affine prediction"])
    # plt.title("$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[1], lambda_2[1]))
    # plt.savefig("lambda_pdf_max_p{}.png".format(p))

    # Pdf stretch def

    i = 0

    axs_min[0, 1].hist2d(reference_stretches[i], current_thetas[i], bins=100, density=True)
    axs_min[0, 1].plot(
        predicted_stretches_deformed[0], current_thetas[i], "ko", markersize=1, alpha=1
    )
    axs_min[0, 1].set_xlabel(r"$\lambda_a$", fontsize=13)
    axs_min[0, 1].set_ylabel(r"$\theta$")
    axs_min[0, 1].legend(["Numerical results", "Affine prediction"])
    # plt.title("$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[0], lambda_2[0]))
    # plt.savefig("lambda_def_pdf_min_p{}.png".format(p))

    i = -1

    axs_max[0, 1].hist2d(reference_stretches[i], current_thetas[i], bins=100, density=True)
    axs_max[0, 1].plot(
        predicted_stretches_deformed[i], current_thetas[i], "ko", markersize=1, alpha=1
    )
    axs_max[0, 1].set_xlabel(r"$\lambda_a$", fontsize=13)
    axs_max[0, 1].set_ylabel(r"$\theta$")
    axs_max[0, 1].legend(["Numerical results", "Affine prediction"])
    # plt.title("$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[1], lambda_2[1]))
    # plt.savefig("lambda_def_pdf_max_p{}.png".format(p))

    # Pdf Chi

    i = 0

    axs_min[1, 0].hist2d(chi[i], initial_theta, bins=100, density=True)
    # plt.plot(predicted_stretches[i],orientations_undef,'ko',markersize = 1)
    axs_min[1, 0].set_xlabel(r"$\lambda_p$", fontsize=13)
    axs_min[1, 0].set_ylabel(r"$Θ$")
    # plt.title("$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[0], lambda_2[0]))
    # plt.savefig("chi_pdf_min_p{}.png".format(p))

    i = -1

    axs_max[1, 0].hist2d(chi[i], initial_theta, bins=100, density=True)
    # plt.plot(predicted_stretches[i],orientations_undef,'ko',markersize = 1)
    axs_max[1, 0].set_xlabel(r"$\lambda_p$", fontsize=13)
    axs_max[1, 0].set_ylabel(r"$Θ$")
    # # plt.title("$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[1], lambda_2[1]))
    # plt.savefig("chi_pdf_max_p{}.png".format(p))

    # Hist chi

    i = 0
    plt.figure()
    axs_min[1, 1].hist(chi[i], bins=100, density=True)
    # plt.plot(predicted_stretches[i],orientations_undef,'ko',markersize = 1)
    axs_min[1, 1].set_xlabel(r"$\lambda_p$", fontsize=13)
    axs_min[1, 1].set_ylabel(r"$p(\lambda_p)$", fontsize=13)
    # plt.title("$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[0], lambda_2[0]))
    # plt.savefig("chi_hist_min_p{}.png".format(p))

    i = -1

    axs_max[1, 1].hist(chi[i], bins=100, density=True)
    # plt.plot(predicted_stretches[i],orientations_undef,'ko',markersize = 1)
    axs_max[1, 1].set_xlabel(r"$\lambda_p$", fontsize=13)
    axs_max[1, 1].set_ylabel(r"$p(\lambda_p)$", fontsize=13)
    # plt.title("$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[1], lambda_2[1]))
    # plt.savefig("chi_hist_max_p{}.png".format(p))

    fig_min.suptitle(
        "$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[0], lambda_2[0]), fontsize=40
    )
    fig_max.suptitle(
        "$(p,\lambda_1,\lambda_2) = ({},{},{})$".format(p, lambda_1[1], lambda_2[1]), fontsize=40
    )

    fig_min.savefig("subplots_min_p{}_d{}_s{}.pdf".format(p, density, seed))
    fig_max.savefig("subplots_max_p{}_d{}_s{}.pdf".format(p, density, seed))
    return
