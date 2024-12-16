# Uses RK23 method to run biaxial stretch deformation on network

# Look at using numba package.

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
import Create_PBC_Network  # noqa
import scipy as sp
import scipy.stats as stats
from scipy.sparse import lil_matrix
from datetime import date

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


def Radau_timestepper_PBC(
    L,
    nodes,
    incidence_matrix,
    initial_lengths,
    edge_corrections,
    lambda_1,
    lambda_2,
):
    # This function defines the right hand side of the ODE system, we pass t into the function as it
    # is required by the scipy package we are using, however it has no affect on the output.
    def scipy_fun(t, y):
        matrix_y = np.reshape(y, (incidence_matrix.shape[1], 2))
        l_j = incidence_matrix.dot(matrix_y)
        l_j_hat = normalise_elements(l_j)
        F_j = (np.sqrt(np.einsum("ij,ij->i", l_j, l_j)) - initial_lengths) / initial_lengths
        product = np.einsum("ij,i->ij", l_j_hat, F_j)
        f_jk = incidence_matrix_transpose.dot(product)
        return -np.reshape(f_jk, 2 * np.shape(f_jk)[0], order="C")

    # Calulates the total elastic energy within the network.
    def energy_calc(y):
        matrix_y = np.reshape(y, (incidence_matrix.shape[1], 2))
        l_j = incidence_matrix.dot(matrix_y)
        u_j = vector_of_magnitudes(l_j) - initial_lengths
        return 0.5 * np.matmul(1 / initial_lengths, np.square(u_j))

    # This computes the matrix components of the network hessian, it is neater to use this function
    # than repeatedly write out the 3 lines.
    def hessian_component(l_j_hat, stretch):
        l_j_outer = np.einsum("i,k", l_j_hat, l_j_hat)
        coefficient = 1 - 1 / stretch
        return l_j_outer - coefficient * (np.eye(2) - l_j_outer)

    # This jacobian can be passed to the radau timestepper to potentially improve its performance,
    # although testing has showed for the networks this script works on the jacobian reduces performance.
    def jacobian(y):
        matrix_y = np.reshape(y, (incidence_matrix.shape[1], 2))
        num_edges, num_nodes = incidence_matrix.shape
        edge_vectors = incidence_matrix.dot(matrix_y) + edge_corrections
        edge_lengths = vector_of_magnitudes(edge_vectors)
        edge_vectors_normalised = normalise_elements(edge_vectors)

        # Define the shape of the hessian, each node is associated with a 2x2 matrix
        hessian = np.zeros((2 * num_nodes, 2 * num_nodes))
        # This first loop computes the upper triangular off-diagonal blocks of the Hessian, as the
        # the hessian is symmetric, we only need to compute the diagonal and the upper-right
        # or lower-left triangle off-diagonal components.
        for edge in range(num_edges):
            i, k = np.nonzero(incidence_matrix[edge, :])[0]

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
    # gives the sparsity structure of the Hessian without having to do the (relatively) expensive
    # computation. Passing this structure to the timestepper dramatically improves performance
    def jac_sparsity_structure(y):
        num_edges, num_nodes = incidence_matrix.shape
        hessian = lil_matrix((2 * num_nodes, 2 * num_nodes))
        for edge in range(num_edges):
            i, k = incidence_matrix[edge, :].nonzero()[1]
            hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = 1
            hessian[2 * k : 2 * k + 2, 2 * i : 2 * i + 2] = 1
        return hessian

    y = dilation_deformation(nodes, lambda_1, lambda_2)
    # We flatten the matrix of node positions to work with the scipy package's timestepper
    y = np.reshape(y, 2 * np.shape(y)[0], order="C")

    # Inside the timestepper we do calculations with just the sparse version of the incidence
    # matrix to improve performance. Outside of the timestepper we do not as the structure is
    # harder to work with and manipulate.
    incidence_matrix_transpose = copy.deepcopy(incidence_matrix.T)

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

            # break loop after modeling is finished
            if sol.status == "finished":
                y_values.append(sol.y)
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
            print("convergence not reached in 100 t")
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
                print("convergence not reached in 200 t")
                break

    ###############
    # t_vals = [0] + t_values
    # energy_values = [energy_calc(y)]
    # norm_values = [np.linalg.norm(scipy_fun(None, y))]
    # for i in range(np.shape(y_values)[0]):
    #     energy_values.append(energy_calc(y_values[i]))
    #     norm_values.append(np.linalg.norm(scipy_fun(None, y_values[i])))

    y_output = np.reshape(y_values[-1], (incidence_matrix.shape[1], 2))
    # if Plot_networks:
    #     try:
    #         Create_PBC_network.ColormapPlot_dilation(
    #             y_output, incidence_matrix, L, lambda_1, lambda_2, initial_lengths
    #         )
    #     except IndexError or ZeroDivisionError or ValueError:
    #         pass

    return y_output


# (
#         t_vals,
#         norm_values,
#         y_output,
#         energy_values,
#     )


def Realisation_dilation(
    L,
    density,
    seed,
    fibre_lengths_multiplier,
    max_lambda_1,
    max_lambda_2,
    num_steps,
    Plot_stress_results=False,
    Plot_networks=False,
):
    realisation_start_time = time.time()
    data = []

    lambda_1_step = (max_lambda_1 - 1) / (num_steps - 1)
    lambda_2_step = (max_lambda_2 - 1) / (num_steps - 1)

    (nodes, edge_corrections, incidence_matrix) = Create_PBC_Network.Create_pbc_Network(
        density, L, seed
    )

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
            Radau_timestepper_PBC(
                L,
                input_nodes,
                incidence_matrix,
                initial_lengths,
                edge_corrections,
                1 + lambda_1_step * i,
                1 + lambda_2_step * i,
            )
        )
        # Each step we provide a guess for the solution at the next step using the previous solution
        input_nodes = invert_dilation(
            data[i - flag_skipped_first_computation],
            1 + lambda_1_step * i,
            1 + lambda_2_step * i,
        )

    print("Total Deformation Computational time = ", time.time() - total_time)

    with open(
        "Dilation_pbc_realisation_{}_{}_{}_{}_{}_{}_{}_{}_{}_".format(
            L,
            density,
            seed,
            fibre_lengths_multiplier,
            max_lambda_1,
            max_lambda_2,
            num_steps,
            Plot_stress_results,
            Plot_networks,
        )
        + str(date.today())
        + ".dat",
        "wb",
    ) as f:
        pickle.dump((data, (nodes, edge_corrections, incidence_matrix)), f)

    print("Total Realisation time =", time.time() - realisation_start_time)

    return (data, (nodes, edge_corrections, incidence_matrix))
