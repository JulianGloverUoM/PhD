# -*- coding: utf-8 -*-
# Resolving a balance of forces on a directed network.
# Currently using the PBC_network script to generate the network, however code
# is agnostic to the method used to generate the network. (not true atm)

######################################
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import time
import os
import sys
import pickle
import scipy as sp
import itertools

file_path = os.path.realpath(__file__)
sys.path.append(file_path)
import PBC_network  # noqa


#############################################################################
#############################################################################

# Applies a shearing to the nodes such that the box containing the nodes is
# smoothly deformed into a parallelogram.


def shear_x(input_nodes, shear_factor):
    num_nodes = np.shape(input_nodes)[0]
    output = np.empty([num_nodes, 2], dtype=float)
    deformation = np.array([[1, shear_factor], [0, 1]])
    for i in range(num_nodes):
        output[i] = deformation.dot(input_nodes[i])
    return output


# Inverts the deformation for the purpose of plotting
# The deformation is linear, so once the system has been evolved to ~ steady
# state we can map the network back to the box shape.


def invert_shear_x(input_edges, shear_factor):
    deformation = np.array([[1, -shear_factor], [0, 1]])
    output = []
    for edge in input_edges:
        output.append([list(deformation.dot(edge[0])), list(deformation.dot(edge[1]))])
    return output


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


def frobenius_norm(input_array):
    return np.sqrt(np.einsum("ij,ij", input_array, input_array))


#############################################################################
#############################################################################


def form_equilibrium_matrix(nodes, incidence_matrix):
    l_j = incidence_matrix.dot(nodes)
    l_j_hat = normalise_elements(l_j)
    Equilibrium_matrix = np.empty((2 * len(nodes), len(l_j)))
    for k in range(len(nodes)):
        for j in range(len(l_j)):
            Equilibrium_matrix[int(2 * k), j] = incidence_matrix[j, k] * l_j_hat[j][0]
            Equilibrium_matrix[int(2 * k + 1), j] = incidence_matrix[j, k] * l_j_hat[j][1]
    return Equilibrium_matrix


def form_equilibrium_boundary(nodes, incidence_matrix, boundary_nodes):
    l_j = incidence_matrix.dot(nodes)
    l_j_hat = normalise_elements(l_j)
    Equilibrium_matrix = np.empty((2 * boundary_nodes, len(l_j)))
    for k in range(boundary_nodes):
        for j in range(len(l_j)):
            Equilibrium_matrix[int(2 * k), j] = incidence_matrix[j, k] * l_j_hat[j][0]
            Equilibrium_matrix[int(2 * k + 1), j] = incidence_matrix[j, k] * l_j_hat[j][1]
    return Equilibrium_matrix


#############################################################################
#############################################################################

# Function to calculate sum of the forces for the unknowns given as y


# def scipy_fun(t, y):
#     matrix_y = np.reshape(y, (np.shape(incidence_matrix)[1], 2))
#     l_j = incidence_matrix.dot(matrix_y)
#     l_j_hat = normalise_elements(l_j)
#     u_j = vector_of_magnitudes(l_j) - vector_of_magnitudes(initial_lengths)
#     F_j = np.multiply(initial_lengths_inv, u_j)
#     product = np.array([np.multiply(l_j_hat[:, 0], F_j), np.multiply(l_j_hat[:, 1], F_j)])
#     f_jk = np.matmul(np.transpose(incidence_matrix), np.transpose(product))
#     f_jk[boundary_nodes:] = 0
#     return -np.reshape(f_jk, 2 * np.shape(f_jk)[0], order="C")


# def boundary_force(y):
#     matrix_y = np.reshape(y, (np.shape(incidence_matrix)[1], 2))
#     l_j = incidence_matrix.dot(matrix_y)
#     l_j_hat = normalise_elements(l_j)
#     u_j = vector_of_magnitudes(l_j) - vector_of_magnitudes(initial_lengths)
#     F_j = np.multiply(initial_lengths_inv, u_j)
#     product = np.array([np.multiply(l_j_hat[:, 0], F_j), np.multiply(l_j_hat[:, 1], F_j)])
#     f_jk = np.matmul(np.transpose(incidence_matrix), np.transpose(product))
#     return f_jk


# def energy_calc(y):
#     matrix_y = np.reshape(y, (np.shape(incidence_matrix)[1], 2))
#     l_j = incidence_matrix.dot(matrix_y)
#     u_j = vector_of_magnitudes(l_j) - vector_of_magnitudes(initial_lengths)
#     return 0.5 * np.matmul(initial_lengths_inv, np.square(u_j))


def calculate_stress_strain(data, strain_stepsize, L):
    num_intervals = len(data)

    force_top = []

    force_bot = []

    force_left = []

    force_right = []

    for i in range(num_intervals):
        force_top.append(data[i][0][0])

        force_bot.append(data[i][0][1])

        force_left.append(data[i][0][2])

        force_right.append(data[i][0][3])

    force_top_sum = [np.sum(item, axis=0) for item in force_top]
    force_bot_sum = [np.sum(item, axis=0) for item in force_bot]
    force_left_sum = [np.sum(item, axis=0) for item in force_left]
    force_right_sum = [np.sum(item, axis=0) for item in force_right]

    p_top = -np.stack(np.array(force_top_sum), axis=0) / L
    p_bot = np.stack(np.array(force_bot_sum), axis=0) / L
    p_left = -np.stack(np.array(force_left_sum), axis=0)
    p_right = -np.stack(np.array(force_right_sum), axis=0)

    for i in range(num_intervals):
        gamma = i * strain_stepsize
        sin = np.sin(np.pi / 2 - gamma)
        cos = np.cos(np.pi / 2 - gamma)
        p_left[i] = np.matmul(np.array([[cos, -sin], [sin, cos]]), p_left[i]) / (
            L * np.sqrt(1 + gamma**2)
        )

    for i in range(num_intervals):
        gamma = i * strain_stepsize
        sin = np.sin(np.pi / 2 + gamma)
        cos = np.cos(np.pi / 2 + gamma)
        p_right[i] = np.matmul(np.array([[cos, sin], [-sin, cos]]), p_right[i]) / (
            L * np.sqrt(1 + gamma**2)
        )

    return (
        [force_top, force_bot, force_left, force_right],
        [force_top_sum, force_bot_sum, force_left_sum, force_right_sum],
        [p_top, p_bot, p_left, p_right],
    )


#############################################################################
#############################################################################

# Calculates the mean edge length of the network, then removes all edges shorter than
# mean_factor*mean.


def remove_short_edges(mean_factor, incidence_matrix, nodes, L):
    edge_lengths = vector_of_magnitudes(incidence_matrix.dot(nodes))
    iterated_matrix = incidence_matrix
    iterated_nodes = nodes
    mean = sum(edge_lengths) / len(edge_lengths)

    for i in range(len(edge_lengths)):
        if edge_lengths[i] < mean_factor * mean:
            iterated_matrix[i, :] = 0

    count = 0
    while True:
        del_edges = []
        del_nodes = []
        # Removes the short edges and any dangling edges created by node trimmming
        for i in range(np.shape(iterated_matrix)[0]):
            if len(np.nonzero(iterated_matrix[i, :])[0]) <= 1:
                del_edges.append(i)
        loop_matrix_1 = np.delete(iterated_matrix, del_edges, 0)

        # cleans up matrix by identifying (non-boundary) dangling edges
        for j in range(np.shape(loop_matrix_1)[1]):
            if len(np.nonzero(loop_matrix_1[:, j])[0]) == 0:
                del_nodes.append(j)
            if len(np.nonzero(loop_matrix_1[:, j])[0]) == 1:
                # Array of boolean values, True means node has coord on boundary
                # which means the node is not part of a dangling edge so dont delete node
                boundary_check = (iterated_nodes[j] % L) == 0
                if not any(boundary_check):
                    del_nodes.append(j)

        iterated_nodes = np.delete(iterated_nodes, del_nodes, 0)
        loop_matrix_2 = np.delete(loop_matrix_1, del_nodes, 1)
        if np.shape(loop_matrix_2) == np.shape(iterated_matrix):
            break
        iterated_matrix = loop_matrix_2
        count += 1
        if count == 1000:
            break
    for i in range(len(iterated_nodes)):
        if any(iterated_nodes[i] % L == 0):
            trim_boundary = i
            break
    return (iterated_matrix, iterated_nodes, trim_boundary)


#############################################################################
#############################################################################

# Restructures the network data of a pbc_network to remove connections across the boundaries


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

            # Removing the node connectivity across the boundary from the incidence matrix
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


def Radau_timestepper(
    L, nodes, incidence_matrix, boundary_nodes, initial_lengths, shear_factor, Plot_networks
):
    def scipy_fun(t, y):
        matrix_y = np.reshape(y, (np.shape(incidence_matrix)[1], 2))
        l_j = sparse_incidence.dot(matrix_y)
        l_j_hat = normalise_elements(l_j)
        F_j = (np.sqrt(np.einsum("ij,ij->i", l_j, l_j)) - initial_lengths) / initial_lengths
        product = np.einsum("ij,i->ij", l_j_hat, F_j)
        f_jk = sparse_incidence_transpose.dot(product)
        f_jk[boundary_nodes:] = 0
        return -np.reshape(f_jk, 2 * np.shape(f_jk)[0], order="C")

    def boundary_force(y):
        matrix_y = np.reshape(y, (np.shape(incidence_matrix)[1], 2))
        l_j = sparse_incidence.dot(matrix_y)
        l_j_hat = normalise_elements(l_j)
        F_j = (np.sqrt(np.einsum("ij,ij->i", l_j, l_j)) - initial_lengths) / initial_lengths
        product = np.einsum("ij,i->ij", l_j_hat, F_j)
        f_jk = sparse_incidence_transpose.dot(product)
        return -f_jk

    def energy_calc(y):
        matrix_y = np.reshape(y, (np.shape(incidence_matrix)[1], 2))
        l_j = sparse_incidence.dot(matrix_y)
        u_j = vector_of_magnitudes(l_j) - initial_lengths
        return 0.5 * np.matmul(1 / initial_lengths, np.square(u_j))

    def hessian_component(l_j_hat, stretch):
        l_j_outer = np.einsum("i,k", l_j_hat, l_j_hat)
        coefficient = 1 - 1 / stretch
        return l_j_outer - coefficient * (np.eye(2) - l_j_outer)

    def jacobian(y):
        matrix_y = np.reshape(y, (np.shape(incidence_matrix)[1], 2))
        num_edges, num_nodes = np.shape(incidence_matrix)
        edge_vectors = sparse_incidence.dot(matrix_y)
        edge_lengths = vector_of_magnitudes(edge_vectors)
        edge_vectors_normalised = normalise_elements(edge_vectors)
        hessian = np.zeros((2 * num_nodes, 2 * num_nodes))
        # This first loop computes the upper triangular off-diagonal blocks of the Hessian.
        for edge in range(num_edges):
            i, k = np.nonzero(incidence_matrix[edge, :])[0]
            # These checks incorporate the Neumann BCs. Note as we are calculating just the upper triangular block, i<k
            # Hence if i is on the boundary k must be as well, if i is not a boundary node then k might be, in which
            # case df_i/dr_k = 0 but df_k/dr_i =/= 0 so we calculate it.
            if i >= boundary_nodes:
                continue
            if k >= boundary_nodes:
                stretch = edge_lengths[edge] / initial_lengths[edge]
                component = -hessian_component(edge_vectors_normalised[edge], stretch)
                hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = component
                continue
            stretch = edge_lengths[edge] / initial_lengths[edge]
            component = -hessian_component(edge_vectors_normalised[edge], stretch)
            hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = component
            hessian[2 * k : 2 * k + 2, 2 * i : 2 * i + 2] = component

        # We now compute the more complex diagonal entries
        for node in range(boundary_nodes):
            edge_indices = np.nonzero(incidence_matrix[:, node])[0]
            i = 2 * node
            hessian[i : i + 2, i : i + 2] = sum(
                [
                    hessian_component(
                        edge_vectors_normalised[edge], edge_lengths[edge] / initial_lengths[edge]
                    )
                    for edge in edge_indices
                ]
            )
        return hessian

    def jac_sparsity_structure(y):
        num_edges, num_nodes = np.shape(incidence_matrix)
        hessian = np.zeros((2 * num_nodes, 2 * num_nodes))
        component = np.ones((2, 2))
        # This first loop computes the upper triangular off-diagonal blocks of the Hessian.
        for edge in range(num_edges):
            i, k = np.nonzero(incidence_matrix[edge, :])[0]
            if i >= boundary_nodes:
                hessian[2 * k : 2 * k + 2, 2 * i : 2 * i + 2] = component
                continue
            hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = component
            hessian[2 * k : 2 * k + 2, 2 * i : 2 * i + 2] = component

        # We now compute the more complex diagonal entries
        for node in range(boundary_nodes):
            i = 2 * node
            hessian[i : i + 2, i : i + 2] = component
        return hessian

    # start_time = time.time()
    # print("Shear factor = ", shear_factor)

    # y[boundary_nodes:] = shear_x(nodes[boundary_nodes:], shear_factor)
    y = shear_x(nodes, shear_factor)
    y = np.reshape(y, 2 * np.shape(y)[0], order="C")

    sparse_incidence = sp.sparse.csc_matrix(incidence_matrix)
    sparse_incidence_transpose = sp.sparse.csr_matrix(np.transpose(incidence_matrix))

    y_values = []
    t_values = []

    jac_structure = jac_sparsity_structure(y)

    y_input = y
    ###############
    # Timestepping

    increasing_energy = False

    i = 0
    while True:
        sol = sp.integrate.Radau(
            scipy_fun,
            i,
            y_input,
            (i + 1),
            max_step=np.inf,
            rtol=1e-05,
            atol=1e-07,
            jac=None,
            jac_sparsity=jac_structure,
        )
        while True:
            # get y_new step state
            sol.step()
            t_values.append(sol.t)
            y_values.append(sol.y)

            # break loop after modeling is finished
            if sol.status == "finished":
                # print("Norm after t = ", i + 1)
                # print(np.linalg.norm(scipy_fun(None, sol.y)))
                break
        if energy_calc(y_values[-1]) > energy_calc(y_input):
            print("energy increasing")
            increasing_energy = True
            break
        if np.linalg.norm(scipy_fun(None, y_values[-1])) < 1e-04:
            break

        y_input = y_values[-1]
        i = i + 1
        if i % 10 == 0:
            print(i)
        if i == 100:
            print("convergence not reached in 100 tau")
            break

    # Some networks contain edges or structures that are stiff and require stricter error
    # tolerances for the RK23 scheme to converge to a mechanical equilibrium.
    # However these stricter error tolerances also increase the computational cost
    # of the scheme to we implement a test to only increase the tolerance when required.
    if increasing_energy:
        while True:
            sol = sp.integrate.Radau(
                scipy_fun,
                i,
                y_input,
                (i + 1),
                max_step=np.inf,
                rtol=1e-13,
                atol=1e-16,
                jac=None,
                jac_sparsity=jac_structure,
            )
            while True:
                # get y_new step state
                sol.step()
                t_values.append(sol.t)
                y_values.append(sol.y)

                # break loop after modeling is finished
                if sol.status == "finished":
                    # print("Norm after t = ", i + 1)
                    # print(np.linalg.norm(scipy_fun(None, sol.y)))
                    break
            if np.linalg.norm(scipy_fun(None, y_values[-1])) < 1e-04:
                break

            y_input = y_values[-1]
            i = i + 1
            if i % 10 == 0:
                print(i)
            if i == 100:
                print("convergence not reached in 100 tau")
                break

    ###############
    t_vals = [0] + t_values
    energy_values = [energy_calc(y)]
    norm_values = [np.linalg.norm(scipy_fun(None, y))]
    for i in range(np.shape(y_values)[0]):
        energy_values.append(energy_calc(y_values[i]))
        norm_values.append(np.linalg.norm(scipy_fun(None, y_values[i])))

    top_nodes = []

    bot_nodes = []

    left_nodes = []

    right_nodes = []

    for i in range(len(nodes[boundary_nodes:])):
        if abs(nodes[i + boundary_nodes][1] - L) <= 1e-15:
            top_nodes.append(i + boundary_nodes)
        if abs(nodes[i + boundary_nodes][1] - 0):
            bot_nodes.append(i + boundary_nodes)
        if abs(nodes[i + boundary_nodes][0] - 0) <= 1e-15:
            left_nodes.append(i + boundary_nodes)
        if abs(nodes[i + boundary_nodes][0] - L) <= 1e-15:
            right_nodes.append(i + boundary_nodes)

    force_all = boundary_force(y_values[-1])

    force_top = np.array([force_all[item] for item in top_nodes])

    force_bot = np.array([force_all[item] for item in bot_nodes])

    force_left = np.array([force_all[item] for item in left_nodes])

    force_right = np.array([force_all[item] for item in right_nodes])

    y_output = np.reshape(y_values[-1], (np.shape(incidence_matrix)[1], 2))
    if Plot_networks:
        try:
            PBC_network.ColormapPlot(y_output, incidence_matrix, L, shear_factor, initial_lengths)
        except IndexError or ZeroDivisionError or ValueError:
            pass
    # print(
    #     "Time taken till relaxation for shear factor =",
    #     shear_factor,
    #     "is",
    #     time.time() - start_time,
    #     "s",
    # )

    return (
        [force_top, force_bot, force_left, force_right],
        [t_vals, norm_values],
        y_output,
        energy_values,
    )


#############################################################################
#############################################################################


# L = 1
# density = 20
# seed = 0
# fibre_lengths_multiplier = 1

# Network = PBC_network.CreateNetwork(density * L**2, L, seed)

# pbc_edges = Network[0][2]
# pbc_nodes = Network[1][2]
# pbc_incidence_matrix = Network[2][1]

# # fig_network = plt.figure()
# # PBC_network.PlotNetwork(pbc_edges, L)
# # plt.savefig("Network_L3_seed2_N81.pdf")

# (
#     nodes,
#     edges,
#     incidence_matrix,
#     boundary_nodes,
#     top_nodes,
#     bot_nodes,
#     left_nodes,
#     right_nodes,
# ) = restructure_PBC_data(pbc_edges, pbc_nodes, pbc_incidence_matrix, L)

# initial_lengths = fibre_lengths_multiplier * vector_of_magnitudes(incidence_matrix.dot(nodes))

# PBC_network.PlotNetwork_2(nodes, incidence_matrix, L, 0)

########################################################
########################################################
# Results/plotting
# (L,density,seed,fibre_lengths_multiplier,strain_stepsize,num_intervals,Plot_stress_results,Plot_networks)


def Single_realisation_23(
    L,
    density,
    seed,
    fibre_lengths_multiplier,
    strain_stepsize,
    num_intervals,
    Plot_stress_results,
    Plot_networks,
):
    data = []

    stresses = []

    Network = PBC_network.CreateNetwork(density * L**2, L, seed)

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
    ) = restructure_PBC_data(pbc_edges, pbc_nodes, pbc_incidence_matrix, L)

    # print("Initial fiber lengths multiplier = ", fibre_lengths_multiplier)

    initial_lengths = fibre_lengths_multiplier * vector_of_magnitudes(incidence_matrix.dot(nodes))

    # initial_lengths[10] = 1.5 * initial_lengths[10]

    total_time = time.time()

    input_nodes = np.copy(nodes)
    flag_skipped_first_computation = 0
    for i in range(num_intervals):
        if i == 0 and fibre_lengths_multiplier == 1:
            flag_skipped_first_computation = -1
            continue
        data.append(
            Radau_timestepper(
                L,
                input_nodes,
                incidence_matrix,
                boundary_nodes,
                initial_lengths,
                (strain_stepsize * i),
                Plot_networks,
            )
        )
        # Each step we provide a guess for the solution at the next step using the previous solution
        input_nodes = shear_x(data[i + flag_skipped_first_computation][-2], -strain_stepsize * i)

    print("Total Computational time = ", time.time() - total_time)

    (p_top, p_bot, p_left, p_right) = calculate_stress_strain(data, strain_stepsize, L)[-1]

    stresses.append([p_top, p_bot, p_left, p_right])

    if Plot_stress_results:
        strains = np.linspace(0, strain_stepsize * (num_intervals - 1), num_intervals)
        if fibre_lengths_multiplier == 1:
            strains = np.delete(strains, 0)

        plt.figure()

        plt.plot(strains, p_top[:, 0])
        plt.plot(strains, p_top[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Top boundary")
        # plt.savefig("Prestress_05_top.pdf")

        plt.figure()

        plt.plot(strains, p_bot[:, 0])
        plt.plot(strains, p_bot[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Bottom boundary")
        # plt.savefig("Prestress_05_bot.pdf")

        plt.figure()

        plt.plot(strains, p_left[:, 0])
        plt.plot(strains, p_left[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Left boundary")
        # plt.savefig("Prestress_05_left.pdf")

        plt.figure()

        plt.plot(strains, p_right[:, 0])
        plt.plot(strains, p_right[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Right boundary")
        # plt.savefig("Prestress_05_right.pdf")
    return (data, [p_top, p_bot, p_left, p_right])


def Strain_range_relisation(
    L, density, seed, fibre_lengths_multiplier, strains, Plot_stress_results, Plot_networks
):
    strain_stepsize = strains[2] - strains[1]
    data = []

    stresses = []

    Network = PBC_network.CreateNetwork(density * L**2, L, seed)

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
    ) = restructure_PBC_data(pbc_edges, pbc_nodes, pbc_incidence_matrix, L)

    # print("Initial fiber lengths multiplier = ", fibre_lengths_multiplier)

    initial_lengths = fibre_lengths_multiplier * vector_of_magnitudes(incidence_matrix.dot(nodes))

    # initial_lengths[10] = 1.5 * initial_lengths[10]

    total_time = time.time()

    input_nodes = np.copy(nodes)

    for i in range(len(strains)):
        data.append(
            Radau_timestepper(
                L,
                input_nodes,
                incidence_matrix,
                boundary_nodes,
                initial_lengths,
                strains[i],
                Plot_networks,
            )
        )
        # The first step generates the initial network in its minimum energy configuration after
        # prestress has been applied. This relaxed network is then deformed when applying strains
        if i == 0 and fibre_lengths_multiplier == 1:
            input_nodes = data[0][-2]

    print("Total Computational time = ", time.time() - total_time)

    (p_top, p_bot, p_left, p_right) = calculate_stress_strain(data, strain_stepsize, L)[-1]

    stresses.append([p_top, p_bot, p_left, p_right])

    if Plot_stress_results:
        plt.figure()

        plt.plot(strains, p_top[:, 0])
        plt.plot(strains, p_top[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Top boundary")
        # plt.savefig("Prestress_05_top.pdf")

        plt.figure()

        plt.plot(strains, p_bot[:, 0])
        plt.plot(strains, p_bot[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Bottom boundary")
        # plt.savefig("Prestress_05_bot.pdf")

        plt.figure()

        plt.plot(strains, p_left[:, 0])
        plt.plot(strains, p_left[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Left boundary")
        # plt.savefig("Prestress_05_left.pdf")

        plt.figure()

        plt.plot(strains, p_right[:, 0])
        plt.plot(strains, p_right[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Right boundary")
        # plt.savefig("Prestress_05_right.pdf")
    return (data, [p_top, p_bot, p_left, p_right])


def Many_realisations_23(
    L, density, fibre_lengths_multiplier, num_realisations, strain_stepsize, num_strain_intervals
):
    output_data = []
    output_plot_data = []

    sigma_plot_data = []

    strains = np.linspace(0, strain_stepsize * (num_strain_intervals - 1), num_strain_intervals)

    sigma_average = []

    stress_averages_top = []
    stress_averages_bot = []
    stress_averages_left = []
    stress_averages_right = []
    for i in range(num_strain_intervals):
        stress_averages_top.append(np.array([0, 0]))
        stress_averages_bot.append(np.array([0, 0]))
        stress_averages_left.append(np.array([0, 0]))
        stress_averages_right.append(np.array([0, 0]))
        sigma_average.append(0)

    for i in range(num_realisations):
        # Compute a single realisation for network generated with random seed i
        Network = PBC_network.CreateNetwork(density * L**2, L, i)

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
        ) = restructure_PBC_data(pbc_edges, pbc_nodes, pbc_incidence_matrix, L)

        # print("Initial fiber lengths multiplier = ", fibre_lengths_multiplier)

        initial_lengths = fibre_lengths_multiplier * vector_of_magnitudes(
            incidence_matrix.dot(nodes)
        )

        (realisation_data, realisation_stress_strains) = Single_realisation_23(
            L,
            density,
            i,
            fibre_lengths_multiplier,
            strain_stepsize,
            num_strain_intervals,
            Plot_stress_results=False,
            Plot_networks=False,
        )

        output_data.append(realisation_data)
        output_plot_data.append(realisation_stress_strains)

        realisation_sigma = []

        for j in range(num_strain_intervals):
            realisation_sigma.append(
                discrete_stress(realisation_data[j][-2], incidence_matrix, initial_lengths, L)
            )
        sigma_plot_data.append(realisation_sigma)

        print("Realisation", i + 1, "of", num_realisations, "computed")

    # Computing the mean stress across all reaisations
    for k in range(num_realisations):
        for j in range(num_strain_intervals):
            stress_averages_top[j] = (
                stress_averages_top[j] + output_plot_data[k][0][j] / num_realisations
            )
            stress_averages_bot[j] = (
                stress_averages_bot[j] + output_plot_data[k][1][j] / num_realisations
            )
            stress_averages_left[j] = (
                stress_averages_left[j] + output_plot_data[k][2][j] / num_realisations
            )
            stress_averages_right[j] = (
                stress_averages_right[j] + output_plot_data[k][3][j] / num_realisations
            )
            sigma_average[j] = sigma_average[j] + sigma_plot_data[k][j] / num_realisations

    plt.figure()
    for i in range(num_realisations):
        plt.plot(
            strains, output_plot_data[i][0][:, 0], alpha=0.2, color="black", label="_nolegend_"
        )
        plt.plot(strains, output_plot_data[i][0][:, 1], alpha=0.2, color="red", label="_nolegend_")

    plt.plot(strains, np.array(stress_averages_top)[:, 0], alpha=1, color="black")
    plt.plot(strains, np.array(stress_averages_top)[:, 1], alpha=1, color="red")
    plt.xlabel(r"Strain, $\gamma$")
    plt.ylabel("Stress")
    plt.legend(["Average Shear Stress", "Normal Stress"])
    plt.title("Top boundary")

    plt.figure()
    for i in range(num_realisations):
        plt.plot(
            strains, output_plot_data[i][1][:, 0], alpha=0.2, color="black", label="_nolegend_"
        )
        plt.plot(strains, output_plot_data[i][1][:, 1], alpha=0.2, color="red", label="_nolegend_")

    plt.plot(strains, np.array(stress_averages_bot)[:, 0], alpha=1, color="black")
    plt.plot(strains, np.array(stress_averages_bot)[:, 1], alpha=1, color="red")
    plt.xlabel(r"Strain, $\gamma$")
    plt.ylabel("Stress")
    plt.legend(["Average Shear Stress", "Normal Stress"])
    plt.title("Bot boundary")

    plt.figure()
    for i in range(num_realisations):
        plt.plot(
            strains, output_plot_data[i][2][:, 0], alpha=0.2, color="black", label="_nolegend_"
        )
        plt.plot(strains, output_plot_data[i][2][:, 1], alpha=0.2, color="red", label="_nolegend_")

    plt.plot(strains, np.array(stress_averages_left)[:, 0], alpha=1, color="black")
    plt.plot(strains, np.array(stress_averages_left)[:, 1], alpha=1, color="red")
    plt.xlabel(r"Strain, $\gamma$")
    plt.ylabel("Stress")
    plt.legend(["Average Shear Stress", "Normal Stress"])
    plt.title("Left boundary")

    plt.figure()
    for i in range(num_realisations):
        plt.plot(
            strains, output_plot_data[i][3][:, 0], alpha=0.2, color="black", label="_nolegend_"
        )
        plt.plot(strains, output_plot_data[i][3][:, 1], alpha=0.2, color="red", label="_nolegend_")

    plt.plot(strains, np.array(stress_averages_right)[:, 0], alpha=1, color="black")
    plt.plot(strains, np.array(stress_averages_right)[:, 1], alpha=1, color="red")
    plt.xlabel(r"Strain, $\gamma$")
    plt.ylabel("Stress")
    plt.legend(["Average Shear Stress", "Normal Stress"])
    plt.title("Right boundary")

    sig_xy = []
    for i in range(num_strain_intervals):
        sig_xy.append(sigma_average[i][0, 1])

    sig_xy_spl = sp.interpolate.UnivariateSpline(strains, sig_xy, k=3)
    derivs_sig = sig_xy_spl.derivative(n=1)

    plt.figure()
    plt.plot(strains, sig_xy_spl(strains), "b")
    plt.plot(strains, sig_xy, "ko")
    plt.ylabel(r"$\frac{\partial\sigma_{xy}}{\partial\gamma}$", fontsize=18)
    plt.xlabel(r"Strain, $\gamma$")

    plt.figure()
    plt.plot(strains, derivs_sig(strains), "b")
    plt.plot(strains, derivs_sig(strains), "ko")
    plt.xlabel(r"Strain, $\gamma$")
    plt.ylabel(r"$\sigma_{xy}$", fontsize=16)

    # Each entry of output_data contains the the force, time, norm, final node positions,
    # and energy calculations for each realisation after each strain.
    # Each entry of output_plot_data contains the stress data for all strains of that realisation.
    return (output_data, output_plot_data, sigma_plot_data)


def affinity_energy(L, density, seed, fibre_lengths_multiplier, strain_stepsize, num_intervals):
    output = []

    Network = PBC_network.CreateNetwork(density * L**2, L, seed)

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
    ) = restructure_PBC_data(pbc_edges, pbc_nodes, pbc_incidence_matrix, L)

    initial_lengths = fibre_lengths_multiplier * vector_of_magnitudes(incidence_matrix.dot(nodes))

    input_nodes = np.copy(nodes)

    energy_affine = []
    for i in range(num_intervals):
        energy_affine.append(0)
        alpha = ((1 + strain_stepsize * i) ** 2 + 1) ** 0.5
        for length in initial_lengths:
            energy_affine[i] += length * (alpha**2 + 2 * alpha + 1) * 0.5

    for i in range(num_intervals):
        output.append(
            Radau_timestepper(
                L,
                input_nodes,
                incidence_matrix,
                boundary_nodes,
                initial_lengths,
                (strain_stepsize * i),
                False,
            )
        )
        # The first step generates the initial network in its minimum energy configuration after
        # prestress has been applied. This relaxed network is then deformed when applying strains
        if i == 0:
            input_nodes = output[0][-2]
    for i in range(num_intervals):
        plt.figure()
        plt.plot(output[i][2][0], output[i][-1] / energy_affine[i])
    return output


def remove_short_realisation(
    L, density, seed, fibre_lengths_multiplier, strain_stepsize, num_intervals, mean_factor
):
    data = []

    Network = PBC_network.CreateNetwork(density * L**2, L, seed)

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
    ) = restructure_PBC_data(pbc_edges, pbc_nodes, pbc_incidence_matrix, L)

    (trim_matrix, trim_nodes, trim_boundary_nodes) = remove_short_edges(
        mean_factor, incidence_matrix, nodes, L
    )
    PBC_network.PlotNetwork_2(trim_nodes, trim_matrix, L, 0)

    print("Size of original system =", np.shape(incidence_matrix))
    print("Size of trimmed system =", np.shape(trim_matrix))
    start_time = time.time()

    input_nodes = np.copy(trim_nodes)

    initial_lengths = fibre_lengths_multiplier * vector_of_magnitudes(trim_matrix.dot(trim_nodes))

    for i in range(num_intervals):
        data.append(
            Radau_timestepper(
                L,
                input_nodes,
                trim_matrix,
                trim_boundary_nodes,
                initial_lengths,
                (strain_stepsize * i),
                True,
            )
        )
        # The first step generates the initial network in its minimum energy configuration after
        # prestress has been applied. This relaxed network is then deformed when applying strains
        if i == 0:
            input_nodes = data[0][-2]

    print("Total Computational time = ", time.time() - start_time)
    return data


def discrete_stress(nodes, incidence_matrix, initial_lengths, L):
    edge_vectors = incidence_matrix.dot(nodes)

    normalised_edges = normalise_elements(edge_vectors)

    l_j = vector_of_magnitudes(edge_vectors)

    F_j = (l_j - initial_lengths) / initial_lengths

    sigma = np.matrix([[0, 0], [0, 0]])

    for index in range(len(initial_lengths)):
        sigma = sigma + F_j[index] * l_j[index] * np.outer(
            normalised_edges[index], normalised_edges[index]
        )

    return sigma / (L**2)


def stress_old(nodes, incidence_matrix, initial_lengths, L):
    edge_vectors = incidence_matrix.dot(nodes)

    normalised_edges = normalise_elements(edge_vectors)

    l_j = vector_of_magnitudes(edge_vectors)

    u_j = l_j - initial_lengths

    sigma = np.matrix([[0, 0], [0, 0]])

    for index in range(len(initial_lengths)):
        sigma = sigma + u_j[index] * np.outer(normalised_edges[index], normalised_edges[index])

    return sigma / (L**2)


def Spline_derivative_plot(input_data, strains, knots, derivative_order, data_name):
    input_spl = sp.interpolate.UnivariateSpline(strains, input_data, k=knots)
    plt.figure()
    plt.plot(strains, input_data, "ro")
    plt.plot(strains, input_spl(strains))
    plt.xlabel(r"Strain, $\gamma$")
    plt.ylabel(data_name)
    plt.savefig("5min1.pdf")

    plt.show()
    input_spl_deriv = input_spl.derivative(n=derivative_order)
    plt.figure()
    derivs = input_spl_deriv(strains)
    plt.plot(strains, derivs)
    plt.plot(strains, derivs, "ko")
    plt.xlabel(r"Strain, $\gamma$")
    plt.ylabel("Order {} derivative of {}".format(derivative_order, data_name))
    plt.savefig("5min2.pdf")
    return


# Orientation angle is calculated as the angle in the range [0,\pi] that an edge makes with the
# positive x axis. This is calculated by taking the dot product of an edge vector with e_1 = [1,0]^T
# And rearanging the dot product formula (mod_pi) to obtain desired result


def Orientation_distribrution(nodes, incidence_matrix):
    edge_vectors = incidence_matrix.dot(nodes)
    orientations_output = []
    for i in range(len(edge_vectors)):
        orientations_output.append(
            np.mod(np.arccos(edge_vectors[i][0] / np.linalg.norm(edge_vectors[i])), np.pi)
        )
    return orientations_output


def Edge_lengths_and_orientations_histogram_plot(
    nodes, incidence_matrix, bins_edges, bins_orientations
):
    orientations = Orientation_distribrution(nodes, incidence_matrix)
    edge_lengths = vector_of_magnitudes(incidence_matrix.dot(nodes))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(edge_lengths, bins=bins_edges, density=True)
    ax2.hist(orientations, bins=bins_orientations, density=True)
    return


def affinity_node_position(initial_nodes, input_nodes, gamma):
    return np.linalg.norm(input_nodes - shear_x(initial_nodes, gamma)) / (
        len(input_nodes) * np.linalg.norm(shear_x(initial_nodes, gamma))
    )


def affinity_displacement(initial_nodes, input_nodes, incidence_matrix, gamma):
    initial_lengths = vector_of_magnitudes(incidence_matrix.dot(initial_nodes))
    affine_nodes = shear_x(initial_nodes, gamma)
    L_affine = incidence_matrix.dot(affine_nodes)
    L_affine_lengths = vector_of_magnitudes(L_affine)
    l_j = incidence_matrix.dot(input_nodes)
    l_j_lengths = vector_of_magnitudes(l_j)
    displacement_affine = L_affine_lengths - initial_lengths
    displacement_actual = l_j_lengths - initial_lengths
    return np.linalg.norm(displacement_actual - displacement_affine) / (
        len(initial_lengths) * np.linalg.norm(displacement_affine)
    )


def affinity_strain(initial_nodes, input_nodes, incidence_matrix, gamma):
    initial_lengths = vector_of_magnitudes(incidence_matrix.dot(initial_nodes))
    affine_nodes = shear_x(initial_nodes, gamma)
    L_affine = incidence_matrix.dot(affine_nodes)
    L_affine_lengths = vector_of_magnitudes(L_affine)
    l_j = incidence_matrix.dot(input_nodes)
    l_j_lengths = vector_of_magnitudes(l_j)
    strain_affine = (L_affine_lengths - initial_lengths) / initial_lengths
    strain_actual = (l_j_lengths - initial_lengths) / initial_lengths
    return np.linalg.norm(strain_actual - strain_affine) / (
        len(initial_lengths) * np.linalg.norm(strain_affine)
    )


def compute_adjacency(incidence_matrix):
    num_edges, num_nodes = np.shape(incidence_matrix)
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for j in range(num_edges):
        pairs = list(itertools.combinations(set(np.nonzero(incidence_matrix[j, :])[0]), 2))
        for element in pairs:
            adjacency_matrix[element] = 1
    return adjacency_matrix + np.transpose(adjacency_matrix)


def hessian_component(l_j_hat, stretch):
    l_j_outer = np.einsum("i,k", l_j_hat, l_j_hat)
    coefficient = 1 - 1 / stretch
    return l_j_outer - coefficient * (np.eye(2) - l_j_outer)


def hessian_computation(nodes, incidence_matrix, initial_lengths):
    num_edges, num_nodes = np.shape(incidence_matrix)
    edge_vectors = incidence_matrix.dot(nodes)
    edge_lengths = vector_of_magnitudes(edge_vectors)
    edge_vectors_normalised = normalise_elements(edge_vectors)
    hessian = np.zeros((2 * num_nodes, 2 * num_nodes))
    eigvals = []
    # This first loop computes the upper triangular off-diagonal blocks of the Hessian.
    for edge in range(num_edges):
        i, k = np.nonzero(incidence_matrix[edge, :])[0]
        stretch = edge_lengths[edge] / initial_lengths[edge]
        hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = -hessian_component(
            edge_vectors_normalised[edge], stretch
        )
        eigvals = eigvals + list(np.linalg.eig(hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2])[0])

    # W skip computing the lower triangular blocks as H_ik=H_ki, and H_ik=H_ik^T
    hessian = hessian + np.transpose(hessian)
    # We now compute the more complex diagonal entries
    for node in range(num_nodes):
        edge_indices = np.nonzero(incidence_matrix[:, node])[0]
        i = 2 * node
        hessian[i : i + 2, i : i + 2] = sum(
            [
                hessian_component(
                    edge_vectors_normalised[edge], edge_lengths[edge] / initial_lengths[edge]
                )
                for edge in edge_indices
            ]
        )
        eigvals = eigvals + list(np.linalg.eig(hessian[i : i + 2, i : i + 2])[0])
    return hessian, sorted(eigvals)


#  Useful bit of code below.

# initial_lengths_data = []
# current_lengths_data= []

# energy_data= []
# stretch_data= []

# data_rho_range= []

# L = 2
# fibre_lengths_multiplier = 1
# for density_variable in range(11):
#     density = 10+density_variable
#     for seed_variable in range(5):
#         Network = PBC_network.CreateNetwork(density * L**2, L, seed_variable)

#         pbc_edges = Network[0][2]
#         pbc_nodes = Network[1][2]
#         pbc_incidence_matrix = Network[2][1]

#         (
#             nodes,
#             edges,
#             incidence_matrix,
#             boundary_nodes,
#             top_nodes,
#             bot_nodes,
#             left_nodes,
#             right_nodes,
#         ) = restructure_PBC_data(pbc_edges, pbc_nodes, pbc_incidence_matrix, L)
#         loop_lengths = vector_of_magnitudes(incidence_matrix.dot(nodes))

#         loop_data = Radau_timestepper(L, nodes, incidence_matrix, boundary_nodes, loop_lengths ,1, False)

#         loop_l_j = vector_of_magnitudes(incidence_matrix.dot(loop_data[-2]))
#         loop_energies = 0.5*np.matmul(1/ loop_lengths, np.square(loop_l_j - loop_lengths))
#         loop_stretches = loop_l_j/loop_lengths

#         initial_lengths_data.append(loop_lengths)
#         current_lengths_data.append(loop_l_j)

#         energy_data.append(loop_energies)
#         stretch_data.append(loop_stretches)

#         data_rho_range.append(loop_data)
