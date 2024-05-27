# Uses RK23 method to run biaxial stretch deformation on network

######################################
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import time
import os
import sys
import pickle

file_path = os.path.realpath(__file__)
sys.path.append(file_path)
import PBC_network  # noqa
import scipy as sp

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


def calculate_stress_strain_stretch(
    data, lambda_1_step, lambda_2_step, num_steps, L, fibre_lengths_multiplier
):
    flag_non_zero_initial_stretch = 0
    if fibre_lengths_multiplier == 1:
        flag_non_zero_initial_stretch = 1
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

    for i in range(num_steps):
        # Compute the side length of the domain and divide forces by the length
        # The added flag*lambda*L accounts for when the data starts at non-zero stretch

        L_x = L * (1 + i * lambda_1_step) + flag_non_zero_initial_stretch * lambda_1_step * L
        L_y = L * (1 + i * lambda_2_step) + flag_non_zero_initial_stretch * lambda_2_step * L
        p_top[i] = p_top[i] / L_x
        p_bot[i] = p_bot[i] / L_x
        p_left[i] = np.array([[0, -1], [1, 0]]).dot(p_left[i]) / L_y
        p_right[i] = np.array([[0, 1], [-1, 0]]).dot(p_right[i]) / L_y

    return (
        [force_top, force_bot, force_left, force_right],
        [force_top_sum, force_bot_sum, force_left_sum, force_right_sum],
        [p_top, p_bot, p_left, p_right],
    )


#############################################################################
#############################################################################


def Radau_timestepper_dilation(
    L, nodes, incidence_matrix, boundary_nodes, initial_lengths, lambda_1, lambda_2, Plot_networks
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

    # Effectively computes the Hessian but with all potential non-zero elements = 1, which
    # gives the sparsity structure of the Hessian without having to do the expensive computation
    def jac_sparsity_structure(y):
        num_edges, num_nodes = np.shape(incidence_matrix)
        hessian = np.zeros((2 * num_nodes, 2 * num_nodes))
        component = np.ones((2, 2))
        for edge in range(num_edges):
            i, k = np.nonzero(incidence_matrix[edge, :])[0]
            if i >= boundary_nodes:
                continue
            if k >= boundary_nodes:
                hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = component
                continue
            hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = component
            hessian[2 * k : 2 * k + 2, 2 * i : 2 * i + 2] = component

        for node in range(boundary_nodes):
            i = 2 * node
            hessian[i : i + 2, i : i + 2] = component
        return hessian

    # start_time = time.time()
    # print("Shear factor = ", shear_factor)

    y = dilation_deformation(nodes, lambda_1, lambda_2)
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
            print("equilibrium achieved")
            break

        y_input = y_values[-1]
        i = i + 1
        if i % 10 == 0:
            print(i)
        if i == 100:
            print("convergence not reached in 100 tau")
            increasing_energy = True
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
            if i == 200:
                print("convergence not reached in 200 tau")
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
        if abs(nodes[i + boundary_nodes][1] - 0) <= 1e-15:
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
            PBC_network.ColormapPlot_dilation(
                y_output, incidence_matrix, L, lambda_1, lambda_2, initial_lengths
            )
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

# Currently does not properly deal with computing stress-strain results, needs abit of work


def Realisation_dilation(
    L,
    density,
    seed,
    fibre_lengths_multiplier,
    max_lambda_1,
    max_lambda_2,
    num_steps,
    Plot_stress_results,
    Plot_networks,
):
    data = []

    lambda_1_step = (max_lambda_1 - 1) / num_steps
    lambda_2_step = (max_lambda_2 - 1) / num_steps

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
    for i in range(num_steps):
        if i == 0 and fibre_lengths_multiplier == 1:
            flag_skipped_first_computation = 1
            continue
        data.append(
            Radau_timestepper_dilation(
                L,
                input_nodes,
                incidence_matrix,
                boundary_nodes,
                initial_lengths,
                1 + lambda_1_step * i,
                1 + lambda_2_step * i,
                Plot_networks,
            )
        )
        # Each step we provide a guess for the solution at the next step using the previous solution
        input_nodes = invert_dilation(
            data[i - flag_skipped_first_computation][-2],
            1 + lambda_1_step * i,
            1 + lambda_2_step * i,
        )

    print("Total Computational time = ", time.time() - total_time)

    (p_top, p_bot, p_left, p_right) = calculate_stress_strain_stretch(
        data, lambda_1_step, lambda_2_step, num_steps, L, fibre_lengths_multiplier
    )[-1]

    stresses.append([p_top, p_bot, p_left, p_right])

    if Plot_stress_results:
        lambda_1 = np.linspace(1, max_lambda_1, num_steps)
        lambda_2 = np.linspace(1, max_lambda_2, num_steps)
        if fibre_lengths_multiplier == 1:
            lambda_1 = np.linspace(1 + lambda_1_step, max_lambda_1, num_steps)
            lambda_1 = np.linspace(1 + lambda_2_step, max_lambda_2, num_steps)
        plt.figure()

        plt.plot(lambda_1, p_top[:, 0])
        plt.plot(lambda_1, p_top[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Top boundary")
        # plt.savefig("Prestress_05_top.pdf")

        plt.figure()

        plt.plot(lambda_1, p_bot[:, 0])
        plt.plot(lambda_1, p_bot[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Bottom boundary")
        # plt.savefig("Prestress_05_bot.pdf")

        plt.figure()

        plt.plot(lambda_2, p_left[:, 0])
        plt.plot(lambda_2, p_left[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Left boundary")
        # plt.savefig("Prestress_05_left.pdf")

        plt.figure()

        plt.plot(lambda_2, p_right[:, 0])
        plt.plot(lambda_2, p_right[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Right boundary")
        # plt.savefig("Prestress_05_right.pdf")
    return (data, [p_top, p_bot, p_left, p_right])


# Jacobian calculation is identical to hessian excepting the df_i/dr_k elements when f_i==0 as a
# consequence of boundary conditions.
def hessian_component(l_j_hat, stretch):
    l_j_outer = np.einsum("i,k", l_j_hat, l_j_hat)
    coefficient = 1 - 1 / stretch
    return l_j_outer - coefficient * (np.eye(2) - l_j_outer)


def jacobian(nodes, incidence_matrix, initial_lengths, boundary_nodes):
    num_edges, num_nodes = np.shape(incidence_matrix)
    edge_vectors = incidence_matrix.dot(nodes)
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


# Orientation angle is calculated as the angle in the range [0,\pi] that an edge makes with the
# positive x axis. This is calculated by taking the dot product of an edge vector with e_1 = [1,0]^T
# And rearanging the dot product formula (mod_pi) to obtain desired result


def Orientation_distribution(nodes, incidence_matrix, Plot_data):
    edge_vectors = incidence_matrix.dot(nodes)
    lengths = vector_of_magnitudes(incidence_matrix.dot(nodes))
    orientations_output = []
    for i in range(len(edge_vectors)):
        orientations_output.append(
            np.mod(np.arccos(edge_vectors[i][0] / np.linalg.norm(edge_vectors[i])), np.pi)
        )
    if Plot_data:
        plt.figure()
        plt.hist(
            orientations_output,
            bins=min(int(np.shape(incidence_matrix)[0] / 10), 40),
            density=True,
            weights=lengths,
        )
        plt.xlabel(r"$\theta$")
        plt.title(r"PDF of fibre orientations")
    return orientations_output


def Edge_lengths_and_orientations_histogram_plot(
    nodes, incidence_matrix, bins_edges, bins_orientations, lambda_1, lambda_2
):
    orientations = Orientation_distribution(nodes, incidence_matrix)
    edge_lengths = vector_of_magnitudes(incidence_matrix.dot(nodes))
    plt.figure()
    plt.hist(edge_lengths, bins=bins_edges, density=True)
    plt.title(
        r"PDF of edge length distribution for $(\lambda_1,\lambda_2)$ = {}".format(
            lambda_1, lambda_2
        )
    )

    plt.figure()
    plt.hist(orientations, bins=bins_orientations, density=True)
    plt.title(
        r"PDF of fibre orientations for $(\lambda_1,\lambda_2)$ = {}".format(lambda_1, lambda_2)
    )
    return


def Stretch_distribution(nodes, incidence_matrix, initial_lengths, num_bins, lambda_1, lambda_2):
    stretches = []
    edge_vectors = incidence_matrix.dot(nodes)
    edge_lengths = vector_of_magnitudes(edge_vectors)
    for i in range(len(edge_lengths)):
        stretches.append(edge_lengths[i] / initial_lengths[i])
    plt.figure()
    plt.hist(stretches, bins=num_bins, density=True)
    plt.title(r"PDF of fibre stretches for $(\lambda_1,\lambda_2)$ = {}".format(lambda_1, lambda_2))
    return


def affinity_strain_stretch(initial_nodes, input_nodes, incidence_matrix, lambda_1, lambda_2):
    initial_lengths = vector_of_magnitudes(incidence_matrix.dot(initial_nodes))
    affine_nodes = dilation_deformation(initial_nodes, lambda_1, lambda_2)
    L_affine = incidence_matrix.dot(affine_nodes)
    L_affine_lengths = vector_of_magnitudes(L_affine)
    l_j = incidence_matrix.dot(input_nodes)
    l_j_lengths = vector_of_magnitudes(l_j)
    strain_affine = (L_affine_lengths - initial_lengths) / initial_lengths
    strain_actual = (l_j_lengths - initial_lengths) / initial_lengths
    return np.linalg.norm(strain_actual - strain_affine) / (
        len(initial_lengths) * np.linalg.norm(strain_affine)
    )


def affine_stretch_calc(orientation, lambda_1, lambda_2):
    return np.sqrt(
        lambda_1**2 * np.cos(orientation) ** 2 + lambda_2**2 * np.sin(orientation) ** 2
    )


def affine_stretch_def_calc(orientation, lambda_1, lambda_2):
    return np.sqrt(
        (lambda_1**2 * lambda_2**2)
        / (lambda_2**2 * np.cos(orientation) ** 2 + lambda_1**2 * np.sin(orientation) ** 2)
    )


def chi_undeformed(stretches, orientations, lambda_1, lambda_2):
    output = []
    for i, item in enumerate(orientations):
        output.append(stretches[i] / affine_stretch_calc(item, lambda_1, lambda_2))
    return output


def chi_deformed(stretches, orientations, lambda_1, lambda_2):
    output = []
    for i, item in enumerate(orientations):
        output.append(stretches[i] / affine_stretch_def_calc(item, lambda_1, lambda_2))
    return output


def affine_theta_calc(lambda_1, lambda_2, initial_thetas):
    return [
        np.arcsin(
            lambda_2
            * np.sin(item)
            / np.sqrt(lambda_1**2 * np.cos(item) ** 2 + lambda_2**2 * np.sin(item) ** 2)
        )
        if 0 <= item <= np.pi / 2
        else np.pi
        - np.arcsin(
            lambda_2
            * np.sin(item)
            / np.sqrt(lambda_1**2 * np.cos(item) ** 2 + lambda_2**2 * np.sin(item) ** 2)
        )
        for item in initial_thetas
    ]
