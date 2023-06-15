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

file_path = os.path.realpath(__file__)
sys.path.append(file_path)
import PBC_network  # noqa
import scipy as sp


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
    return np.array([np.array(vector) / np.linalg.norm(vector) for vector in input_array])


# Takes in a array and outputs the magnitudes of the rows in the array


def vector_of_magnitudes(input_array):
    return np.array([np.linalg.norm(vector) for vector in input_array])


def vector_of_inv_magnitudes(input_array):
    return np.array([np.linalg.norm(vector) ** (-1) for vector in input_array])


#############################################################################
#############################################################################

# Function to calculate sum of the forces for the unknowns given as y


# def scipy_fun(t, y):
#     matrix_y = np.reshape(y, (np.shape(adjacency_matrix)[1], 2))
#     l_j = adjacency_matrix.dot(matrix_y)
#     l_j_hat = normalise_elements(l_j)
#     u_j = vector_of_magnitudes(l_j) - vector_of_magnitudes(initial_lengths)
#     F_j = np.multiply(initial_lengths_inv, u_j)
#     product = np.array([np.multiply(l_j_hat[:, 0], F_j), np.multiply(l_j_hat[:, 1], F_j)])
#     f_jk = np.matmul(np.transpose(adjacency_matrix), np.transpose(product))
#     f_jk[boundary_nodes:] = 0
#     return -np.reshape(f_jk, 2 * np.shape(f_jk)[0], order="C")


# def boundary_force(y):
#     matrix_y = np.reshape(y, (np.shape(adjacency_matrix)[1], 2))
#     l_j = adjacency_matrix.dot(matrix_y)
#     l_j_hat = normalise_elements(l_j)
#     u_j = vector_of_magnitudes(l_j) - vector_of_magnitudes(initial_lengths)
#     F_j = np.multiply(initial_lengths_inv, u_j)
#     product = np.array([np.multiply(l_j_hat[:, 0], F_j), np.multiply(l_j_hat[:, 1], F_j)])
#     f_jk = np.matmul(np.transpose(adjacency_matrix), np.transpose(product))
#     return f_jk


# def energy_calc(y):
#     matrix_y = np.reshape(y, (np.shape(adjacency_matrix)[1], 2))
#     l_j = adjacency_matrix.dot(matrix_y)
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

    force_top_sum = force_top_sum
    force_bot_sum = force_bot_sum
    force_left_sum = force_left_sum
    force_right_sum = force_right_sum

    p_top = np.stack(np.array(force_top_sum), axis=0) / L
    p_bot = np.stack(np.array(force_bot_sum), axis=0) / L
    p_left = np.stack(np.array(force_left_sum), axis=0)
    p_right = np.stack(np.array(force_right_sum), axis=0)

    for i in range(num_intervals):
        s = i * strain_stepsize
        sin = 1 / np.sqrt(1 + s * s)
        cos = s * sin
        p_left[i] = np.matmul(np.array([[cos, sin], [-sin, cos]]), p_left[i]) * sin / L

    for i in range(num_intervals):
        s = i * strain_stepsize
        sin = 1 / np.sqrt(1 + s * s)
        cos = s * sin
        p_right[i] = np.matmul(np.array([[-cos, -sin], [sin, -cos]]), p_right[i]) * sin / L

    p_top[:, 1] = -p_top[:, 1]
    p_bot = -p_bot
    return (
        [force_top, force_bot, force_left, force_right],
        [force_top_sum, force_bot_sum, force_left_sum, force_right_sum],
        [p_top, p_bot, p_left, p_right],
    )


#############################################################################
#############################################################################

# Restructures the network data of a pbc_network to remove connections across the boundaries


def restructure_PBC_data(pbc_edges, pbc_nodes, pbc_adjacency_matrix, L):
    PBC_data = []

    for edge in pbc_edges:
        if len(edge) == 2:
            PBC_data.append(
                [
                    pbc_edges.index(edge),
                    np.nonzero(pbc_adjacency_matrix[pbc_edges.index(edge)])[0],
                    edge[0][1],
                    edge[1][0],
                ]
            )
        if len(edge) == 3:
            PBC_data.append(
                [
                    pbc_edges.index(edge),
                    np.nonzero(pbc_adjacency_matrix[pbc_edges.index(edge)])[0],
                    edge[0][1],
                    edge[1],
                    edge[2][0],
                ]
            )

    adjacency_matrix = np.copy(pbc_adjacency_matrix)

    edges = copy.deepcopy(pbc_edges)
    nodes = np.copy(pbc_nodes)

    adjacency_matrix = np.block(
        [
            [adjacency_matrix, np.zeros([len(pbc_edges), 2 * len(PBC_data)])],
            [np.zeros([len(PBC_data), 2 * len(PBC_data) + np.shape(adjacency_matrix)[1]])],
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
            node_indices = np.flip(node_indices)
        if len(item) == 4:
            # Truncating old boundary edges and adding 'new' edges by
            # adding the end segment as a new edge, and remove that segement (and midde part)
            edges.append([edges[edge_index][-1]])
            edges[edge_index].remove(edges[edge_index][-1])

            # Removing the node connectivity across the boundary from the adjacency matrix
            # and replacing it with connections to 'new' nodes added on the boundary
            adjacency_matrix[edge_index] = 0
            adjacency_matrix[edge_index][node_indices[0]] = 1
            adjacency_matrix[edge_index][len(nodes)] = -1
            nodes = np.append(nodes, np.array([item[2]]), axis=0)

            adjacency_matrix[len(edges) - 1] = 0
            adjacency_matrix[len(edges) - 1][node_indices[1]] = 1
            adjacency_matrix[len(edges) - 1][len(nodes)] = -1
            nodes = np.append(nodes, np.array([item[3]]), axis=0)
        else:
            edges.append([edges[edge_index][-1]])
            edges[edge_index].remove(edges[edge_index][-1])
            edges[edge_index].remove(edges[edge_index][-1])

            adjacency_matrix[edge_index] = 0
            adjacency_matrix[edge_index][node_indices[0]] = 1
            adjacency_matrix[edge_index][len(nodes)] = -1
            nodes = np.append(nodes, np.array([item[2]]), axis=0)

            adjacency_matrix[len(edges) - 1] = 0
            adjacency_matrix[len(edges) - 1][node_indices[1]] = 1
            adjacency_matrix[len(edges) - 1][len(nodes)] = -1
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
        adjacency_matrix,
        boundary_nodes,
        top_nodes,
        bot_nodes,
        left_nodes,
        right_nodes,
    )


#############################################################################
#############################################################################


def RK45_timestepper(
    L, nodes, adjacency_matrix, boundary_nodes, initial_lengths, shear_factor, Plot_networks
):
    def scipy_fun(t, y):
        matrix_y = np.reshape(y, (np.shape(adjacency_matrix)[1], 2))
        l_j = adjacency_matrix.dot(matrix_y)
        l_j_hat = normalise_elements(l_j)
        u_j = vector_of_magnitudes(l_j) - vector_of_magnitudes(initial_lengths)
        F_j = np.multiply(initial_lengths_inv, u_j)
        product = np.array([np.multiply(l_j_hat[:, 0], F_j), np.multiply(l_j_hat[:, 1], F_j)])
        f_jk = np.matmul(np.transpose(adjacency_matrix), np.transpose(product))
        f_jk[boundary_nodes:] = 0
        return -np.reshape(f_jk, 2 * np.shape(f_jk)[0], order="C")

    def boundary_force(y):
        matrix_y = np.reshape(y, (np.shape(adjacency_matrix)[1], 2))
        l_j = adjacency_matrix.dot(matrix_y)
        l_j_hat = normalise_elements(l_j)
        u_j = vector_of_magnitudes(l_j) - vector_of_magnitudes(initial_lengths)
        F_j = np.multiply(initial_lengths_inv, u_j)
        product = np.array([np.multiply(l_j_hat[:, 0], F_j), np.multiply(l_j_hat[:, 1], F_j)])
        f_jk = np.matmul(np.transpose(adjacency_matrix), np.transpose(product))
        return f_jk

    def energy_calc(y):
        matrix_y = np.reshape(y, (np.shape(adjacency_matrix)[1], 2))
        l_j = adjacency_matrix.dot(matrix_y)
        u_j = vector_of_magnitudes(l_j) - vector_of_magnitudes(initial_lengths)
        return 0.5 * np.matmul(initial_lengths_inv, np.square(u_j))

    # start_time = time.time()
    # print("Shear factor = ", shear_factor)

    # y[boundary_nodes:] = shear_x(nodes[boundary_nodes:], shear_factor)
    y = shear_x(nodes, shear_factor)
    y = np.reshape(y, 2 * np.shape(y)[0], order="C")

    initial_lengths_inv = np.reciprocal(initial_lengths)

    y_values = []
    t_values = []

    max_dt = 1e-03

    y_input = y
    ###############
    # Timestepping

    increasing_norms = False

    i = 0
    while True:
        sol = sp.integrate.RK45(
            scipy_fun,
            i,
            y_input,
            (i + 1),
            max_dt,
            rtol=0.001,
            atol=1e-06,
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
        if np.linalg.norm(scipy_fun(None, y_values[-1])) > np.linalg.norm(scipy_fun(None, y_input)):
            print("norms increasing")
            increasing_norms = True
            break
        if np.linalg.norm(scipy_fun(None, y_values[-1])) < 1e-04:
            break

        y_input = y_values[-1]
        i = i + 1
        if i == 100:
            break

    # Some networks contain edges or structures that are stiff and require stricter error
    # tolerances for the RK45 scheme to converge to a mechanical equilibrium.
    # However these stricter error tolerances also increase the computational cost
    # of the scheme to we implement a test to only decrease the tolerance when required.
    if increasing_norms:

        y_values = []
        t_values = []

        y_input = y
        while True:
            sol = sp.integrate.RK45(
                scipy_fun,
                i,
                y_input,
                (i + 1),
                max_dt,
                rtol=1e-13,
                atol=1e-16,
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
            if i == 100:
                break

    ###############
    t_vals = [0] + t_values
    energy_values = [np.log(energy_calc(y))]
    norm_values = [np.log(np.linalg.norm(scipy_fun(None, y)))]
    for i in range(np.shape(y_values)[0]):
        energy_values.append(np.log(energy_calc(y_values[i])))
        norm_values.append(np.log(np.linalg.norm(scipy_fun(None, y_values[i]))))

    # fig_energy = plt.figure()
    # plt.plot(t_vals, energy_values)

    # plt.xlabel("dimensionless time, t")
    # plt.ylabel("log(Total Energy)")
    # plt.title(str("shear factor = {}%".format(100 * shear_factor)))

    # fig_norms = plt.figure()
    # plt.plot(t_vals, norm_values)

    # plt.xlabel("dimensionless time, t")
    # plt.ylabel("log(Norms)")
    # plt.title(str("shear factor = {}%".format(100 * shear_factor)))

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

    force_all = boundary_force(y_values[-1])

    force_top = np.array([force_all[item] for item in top_nodes])

    force_bot = np.array([force_all[item] for item in bot_nodes])

    force_left = np.array([force_all[item] for item in left_nodes])

    force_right = np.array([force_all[item] for item in right_nodes])

    y_output = np.reshape(y_values[-1], (np.shape(adjacency_matrix)[1], 2))
    if Plot_networks:
        try:
            PBC_network.ColormapPlot(
                y_output, adjacency_matrix, L, shear_factor, initial_lengths, initial_lengths_inv
            )
        except IndexError or ZeroDivisionError:
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
        t_values,
        [t_vals, norm_values],
        y_output,
        energy_values,
    )


# L = 1
# # Roughly ensures the network will be percolated, as percolation threshold is crossed for
# # > 6.7 fibres per unit area of the box i.e: N/L^2 > 6.7
# N = 10 * L**2

# Network = PBC_network.CreateNetwork(N, L, 7)

# pbc_edges = Network[0][2]
# pbc_nodes = Network[1][2]
# pbc_adjacency_matrix = Network[2][1]

# # fig_network = plt.figure()
# # PBC_network.PlotNetwork(pbc_edges, L)
# # plt.savefig("Network_L3_seed2_N81.pdf")

# (
#     nodes,
#     edges,
#     adjacency_matrix,
#     boundary_nodes,
#     top_nodes,
#     bot_nodes,
#     left_nodes,
#     right_nodes,
# ) = restructure_PBC_data(pbc_edges, pbc_nodes, pbc_adjacency_matrix,L)

# (num_edges, num_nodes) = np.shape(adjacency_matrix)

# PBC_network.PlotNetwork_2(nodes, adjacency_matrix, L, 0)


# L = 1
# # Roughly ensures the network will be percolated, as percolation threshold is crossed for
# # > 6.7 fibres per unit area of the box i.e: N/L^2 > 6.7
# N = 10 * L**2

# Network = PBC_network.CreateNetwork(N, L, 7)

# pbc_edges = Network[0][2]
# pbc_nodes = Network[1][2]
# pbc_adjacency_matrix = Network[2][1]

# plt.figure(0)
# PBC_network.PlotNetwork(pbc_edges, L)
# plt.plot([0.3, 0.5, 0.4, 0.3], [0.8, 0.8, 0.9, 0.8], color="black")
# plt.savefig("Hanging_cluster.pdf")
########################################################
########################################################
# Results/plotting


def Single_realisation(
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

    Network = PBC_network.CreateNetwork(density * L**2, L, 7)

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
    ) = restructure_PBC_data(pbc_edges, pbc_nodes, pbc_adjacency_matrix, L)

    # print("Initial fiber lengths multiplier = ", fibre_lengths_multiplier)

    initial_lengths = fibre_lengths_multiplier * vector_of_magnitudes(adjacency_matrix.dot(nodes))

    # initial_lengths[10] = 1.5 * initial_lengths[10]

    total_time = time.time()

    input_nodes = np.copy(nodes)

    for i in range(num_intervals):
        data.append(
            RK45_timestepper(
                L,
                input_nodes,
                adjacency_matrix,
                boundary_nodes,
                initial_lengths,
                (strain_stepsize * i),
                Plot_networks,
            )
        )
        # The first step generates the initial network in its minimum energy configuration after
        # prestress has been applied. This relaxed network is then deformed when applying strains
        if i == 0:
            input_nodes = data[0][-2]

    print("Total Computational time = ", time.time() - total_time)

    (p_top, p_bot, p_left, p_right) = calculate_stress_strain(data, strain_stepsize, L)[-1]

    if Plot_stress_results:
        strains = np.linspace(0, 100 * strain_stepsize * (num_intervals - 1), num_intervals)

        plt.figure(0)

        plt.plot(strains, p_top[:, 0])
        plt.plot(strains, p_top[:, 1])
        plt.xlabel("Strain %")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Top boundary")
        # plt.savefig("Prestress_05_top.pdf")

        plt.figure(1)

        plt.plot(strains, p_bot[:, 0])
        plt.plot(strains, p_bot[:, 1])
        plt.xlabel("Strain %")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Bottom boundary")
        # plt.savefig("Prestress_05_bot.pdf")

        plt.figure(2)

        plt.plot(strains, p_left[:, 0])
        plt.plot(strains, p_left[:, 1])
        plt.xlabel("Strain %")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Left boundary")
        # plt.savefig("Prestress_05_left.pdf")

        plt.figure(3)

        plt.plot(strains, p_right[:, 0])
        plt.plot(strains, p_right[:, 1])
        plt.xlabel("Strain %")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Right boundary")
        # plt.savefig("Prestress_05_right.pdf")
    return (data, [p_top, p_bot, p_left, p_right])


def Many_realisations(
    L, density, fibre_lengths_multiplier, num_realisations, strain_stepsize, num_strain_intervals
):
    output_data = []
    output_plot_data = []

    strains = np.linspace(
        0, 100 * strain_stepsize * (num_strain_intervals - 1), num_strain_intervals
    )

    stress_averages_top = []
    stress_averages_bot = []
    stress_averages_left = []
    stress_averages_right = []
    for i in range(num_strain_intervals):
        stress_averages_top.append(np.array([0, 0]))
        stress_averages_bot.append(np.array([0, 0]))
        stress_averages_left.append(np.array([0, 0]))
        stress_averages_right.append(np.array([0, 0]))

    for i in range(num_realisations):
        # Compute a single realisation for network generated with random seed i

        (realisation_data, realisation_stress_strains) = Single_realisation(
            L,
            density,
            i,
            fibre_lengths_multiplier,
            strain_stepsize,
            num_strain_intervals,
            Plot_stress_results=True,
            Plot_networks=True,
        )
        output_data.append(realisation_data)
        output_plot_data.append(realisation_stress_strains)

        (p_top, p_bot, p_left, p_right) = realisation_stress_strains

        # plt.figure(0)

        # plt.plot(strains, p_top[:, 0], alpha=0.4, color="blue", label="_nolegend_")
        # plt.plot(strains, p_top[:, 1], alpha=0.4, color="orange")

        # plt.figure(1)

        # plt.plot(strains, p_bot[:, 0], alpha=0.4, color="blue", label="_nolegend_")
        # plt.plot(strains, p_bot[:, 1], alpha=0.4, color="orange", label="_nolegend_")

        # plt.figure(2)

        # plt.plot(strains, p_left[:, 0], alpha=0.4, color="blue", label="_nolegend_")
        # plt.plot(strains, p_left[:, 1], alpha=0.4, color="orange", label="_nolegend_")

        # plt.figure(3)

        # plt.plot(strains, p_right[:, 0], alpha=0.4, color="blue", label="_nolegend_")
        # plt.plot(strains, p_right[:, 1], alpha=0.4, color="orange", label="_nolegend_")

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

    plt.figure(0)

    plt.plot(strains, np.array(stress_averages_top)[:, 0], alpha=1, color="black")
    plt.plot(strains, np.array(stress_averages_top)[:, 1], alpha=1, color="red")
    plt.xlabel("Strain %")
    plt.ylabel("Stress")
    plt.legend(["Average Shear Stress", "Normal Stress"])
    plt.title("Top boundary")

    plt.figure(1)

    plt.plot(strains, np.array(stress_averages_bot)[:, 0], alpha=1, color="black")
    plt.plot(strains, np.array(stress_averages_bot)[:, 1], alpha=1, color="red")
    plt.xlabel("Strain %")
    plt.ylabel("Stress")
    plt.legend(["Average Shear Stress", "Normal Stress"])
    plt.title("Bottom boundary")

    plt.figure(2)

    plt.plot(strains, np.array(stress_averages_left)[:, 0], alpha=1, color="black")
    plt.plot(strains, np.array(stress_averages_left)[:, 1], alpha=1, color="red")
    plt.xlabel("Strain %")
    plt.ylabel("Stress")
    plt.legend(["Average Shear Stress", "Normal Stress"])
    plt.title("Left boundary")

    plt.figure(3)

    plt.plot(strains, np.array(stress_averages_right)[:, 0], alpha=1, color="black")
    plt.plot(strains, np.array(stress_averages_right)[:, 1], alpha=1, color="red")
    plt.xlabel("Strain %")
    plt.ylabel("Stress")
    plt.legend(["Average Shear Stress", "Normal Stress"])
    plt.title("Right boundary")

    # Each entry of output_data contains the the force, time, norm, final node positions,
    # and energy calculations for eacg realisation after each strain.
    # Each entry of output_plot_data contains the stress data for all strains of that realisation.
    return (output_data, output_plot_data)


def affinity_energy(L, density, seed, fibre_lengths_multiplier, strain_stepsize, num_intervals):

    output = []

    Network = PBC_network.CreateNetwork(density * L**2, L, 7)

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
    ) = restructure_PBC_data(pbc_edges, pbc_nodes, pbc_adjacency_matrix, L)

    initial_lengths = fibre_lengths_multiplier * vector_of_magnitudes(adjacency_matrix.dot(nodes))

    input_nodes = np.copy(nodes)

    energy_affine = []
    for i in range(num_intervals):
        energy_affine.append(0)
        alpha = ((1 + strain_stepsize * i) ** 2 + 1) ** 0.5
        for length in initial_lengths:
            energy_affine[i] += length * (alpha**2 + 2 * alpha + 1) * 0.5

    for i in range(num_intervals):
        output.append(
            RK45_timestepper(
                L,
                input_nodes,
                adjacency_matrix,
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
        plt.plot(output[i][2][0], output[i][-1] / energy_affine[i])
    return output
