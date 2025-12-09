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
import Fixed_BC_script  # noqa
import scipy as sp
from scipy.sparse import lil_matrix
import scipy.stats as stats
from datetime import date

#############################################################################
#############################################################################

# Applies homogenous stretch/dilation deformation to Nx2 array of N nodes


def dilation_deformation(input_nodes, Lambda_1, Lambda_2):
    return input_nodes * np.array([Lambda_1, Lambda_2])


def invert_dilation(input_nodes, Lambda_1, Lambda_2):
    return np.array(
        [np.array([[1 / Lambda_1, 0], [0, 1 / Lambda_2]]).dot(item) for item in input_nodes]
    )


#############################################################################
#############################################################################

# Normalises the rows of an array


# def normalise_elements(input_array):
#     return np.array(
#         [(np.array(vector) / np.sqrt(np.einsum("i,i", vector, vector))) for vector in input_array]
#     )


def normalise_elements(A):
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / norms


# Takes in a array and outputs the magnitudes of the rows in the array


def vector_of_magnitudes(A):
    return np.linalg.norm(A, axis=1)


# Computes the forbenius norm of a matrix, more efficient than inbuilt np.linalg.norm function


def frobenius_norm(A):
    return np.linalg.norm(A)


#############################################################################
#############################################################################


def calculate_stress_strain_stretch(
    data, Lambda_1_step, Lambda_2_step, num_steps, L, fibre_lengths_multiplier
):
    flag_non_zero_initial_stretch = 0
    if fibre_lengths_multiplier != 1:
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

    if not flag_non_zero_initial_stretch:
        force_top_sum = [np.array([0, 0])] + force_top_sum
        force_bot_sum = [np.array([0, 0])] + force_bot_sum
        force_left_sum = [np.array([0, 0])] + force_left_sum
        force_right_sum = [np.array([0, 0])] + force_right_sum

    p_top = -np.stack(np.array(force_top_sum), axis=0)
    p_bot = np.stack(np.array(force_bot_sum), axis=0)
    p_left = -np.stack(np.array(force_left_sum), axis=0)
    p_right = -np.stack(np.array(force_right_sum), axis=0)

    for i in range(num_steps):
        # Compute the side length of the domain and divide forces by the length
        # The added flag*lambda*L accounts for when the data starts at non-zero stretch

        L_x = L * (1 + i * Lambda_1_step) + flag_non_zero_initial_stretch * Lambda_1_step * L
        L_y = L * (1 + i * Lambda_2_step) + flag_non_zero_initial_stretch * Lambda_2_step * L
        p_top[i] = p_top[i] / L_x
        p_bot[i] = p_bot[i] / L_x
        p_left[i] = np.array([[0, 1], [-1, 0]]).dot(p_left[i]) / L_y
        p_right[i] = np.array([[0, -1], [1, 0]]).dot(p_right[i]) / L_y

    return (
        [force_top, force_bot, force_left, force_right],
        [force_top_sum, force_bot_sum, force_left_sum, force_right_sum],
        [p_top, p_bot, p_left, p_right],
    )


#############################################################################
#############################################################################


def Radau_timestepper_dilation(
    L,
    nodes,
    incidence_matrix,
    boundary_nodes,
    initial_lengths,
    Lambda_1,
    Lambda_2,
    Plot_networks=False,
):
    num_nodes = incidence_matrix.shape[1]
    incidence_T = incidence_matrix.T
    inv_initial_lengths = 1.0 / initial_lengths

    def scipy_fun(t, y):
        matrix_y = y.reshape(num_nodes, 2)
        l_j = incidence_matrix.dot(matrix_y)
        l_j_lengths = np.linalg.norm(l_j, axis=1)
        l_j_hat = l_j / l_j_lengths[:, None]
        F_j = l_j_lengths * inv_initial_lengths - 1.0
        product = l_j_hat * F_j[:, None]
        f_jk = incidence_matrix.T.dot(product)
        f_jk[boundary_nodes:] = 0
        return -f_jk.ravel(order="C")

    def boundary_force(y):
        matrix_y = np.reshape(y, (np.shape(incidence_matrix)[1], 2))
        l_j = incidence_matrix.dot(matrix_y)
        l_j_hat = normalise_elements(l_j)
        F_j = (np.sqrt(np.einsum("ij,ij->i", l_j, l_j)) - initial_lengths) / initial_lengths
        product = np.einsum("ij,i->ij", l_j_hat, F_j)
        f_jk = incidence_matrix.T.dot(product)
        return -f_jk

    def energy_calc(y):
        matrix_y = np.reshape(y, (np.shape(incidence_matrix)[1], 2))
        l_j = incidence_matrix.dot(matrix_y)
        u_j = vector_of_magnitudes(l_j) - initial_lengths
        return 0.5 * np.matmul(1 / initial_lengths, np.square(u_j))

    def jacobian(t, y):
        matrix_y = np.reshape(y, (np.shape(incidence_matrix)[1], 2))
        num_edges, num_nodes = np.shape(incidence_matrix)
        hessian = lil_matrix((2 * num_nodes, 2 * num_nodes))

        l_j = incidence_matrix.dot(matrix_y)
        l_j_lengths = np.linalg.norm(l_j, axis=1)
        l_j_hat = l_j / l_j_lengths[:, None]
        l_j_hat_outer_products = l_j_hat[:, :, None] * l_j_hat[:, None, :]

        stretches = l_j / initial_lengths[:, None]
        I_2 = np.eye(2)
        coefficients = 1 - 1 / stretches
        hessian_components = -(
            l_j_hat_outer_products - coefficients[:, None, None] * (I_2 - l_j_hat_outer_products)
        )

        # This first loop computes the upper triangular off-diagonal blocks of the Hessian.
        for edge in range(num_edges):
            i, k = incidence_matrix.getrow(edge).indices
            # These checks incorporate the Neumann BCs. Note as we are calculating just the upper triangular block, i<k
            # Hence if i is on the boundary k must be as well, if i is not a boundary node then k might be, in which
            # case df_i/dr_k = 0 but df_k/dr_i =/= 0 so we calculate it.
            if i >= boundary_nodes:
                continue
            if k >= boundary_nodes:
                hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = hessian_components[edge, edge]
                continue
            else:
                hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = hessian_components[edge, edge]
                hessian[2 * k : 2 * k + 2, 2 * i : 2 * i + 2] = hessian_components[edge, edge]

        # We now compute the more complex diagonal entries
        for node in range(boundary_nodes):
            edge_indices = np.nonzero(incidence_matrix[:, node])[0]
            i = 2 * node
            hessian[i : i + 2, i : i + 2] = sum(
                [hessian_components[edge, edge] for edge in edge_indices]
            )
        return hessian

    # Effectively computes the Hessian but with all potential non-zero elements = 1, which
    # gives the sparsity structure of the Hessian without having to do the expensive computation
    def jac_sparsity_structure(y):
        num_edges, num_nodes = np.shape(incidence_matrix)
        hessian = lil_matrix((2 * num_nodes, 2 * num_nodes))
        component = np.ones((2, 2))
        # if two edges are incident and one isnt a boundary node, make its local component's non-zero
        for edge in range(num_edges):
            i, k = incidence_matrix.getrow(edge).indices
            if i >= boundary_nodes:
                continue
            if k >= boundary_nodes:
                hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = component
                continue
            hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = component
            hessian[2 * k : 2 * k + 2, 2 * i : 2 * i + 2] = component
        # Make all diagonal entries non-zero, except the boundary nodes
        for node in range(boundary_nodes):
            i = 2 * node
            hessian[i : i + 2, i : i + 2] = component
        return hessian

    # start_time = time.time()
    # print("Shear factor = ", shear_factor)

    y = dilation_deformation(nodes, Lambda_1, Lambda_2)
    y = np.reshape(y, 2 * np.shape(y)[0], order="C")

    # mass_vector = 0.5 * abs(incidence_matrix.T).dot(initial_lengths)

    y_vals = []
    t_vals = []

    jac_structure = jac_sparsity_structure(y)

    y_input = y
    ###############
    # Timestepping

    increasing_energy = False
    slow_convergence = False

    max_tau = 200.0  # or whatever “upper bound” you’d allow

    sol = sp.integrate.Radau(
        scipy_fun,
        0.0,
        y,
        max_tau,
        max_step=np.inf,
        rtol=1e-6,
        atol=1e-6,
        jac=None,
        jac_sparsity=jac_structure,
    )

    t_vals = [0.0]
    y_vals = [y.copy()]

    while True:
        sol.step()
        if sol.status in ("finished", "failed"):
            print("Equilibrium achieved/failed")
            break

        if len(y_vals) > 2 and energy_calc(sol.y) > energy_calc(y_vals[-2]):
            print("Energy increasing")
            increasing_energy = True
            break

        t_vals.append(sol.t)
        y_vals.append(sol.y.copy())

        # equilibrium / energy tests
        if sol.t >= 100:
            print("Slow convergence")
            slow_convergence = True
            break
        if (
            max(
                vector_of_magnitudes(
                    np.reshape(scipy_fun(None, sol.y), (np.shape(incidence_matrix)[1], 2))
                )
            )
            < 1e-4
        ):
            print("Equilibrium achieved")
            break

    # Some networks contain edges or structures that are stiff and require stricter error
    # tolerances for the RK23 scheme to converge to a mechanical equilibrium.
    # However these stricter error tolerances also increase the computational cost
    # of the scheme to we implement a test to only increase the tolerance when required.
    if increasing_energy or slow_convergence:
        sol = sp.integrate.Radau(
            scipy_fun,
            t_vals[-1],
            y_vals[-1],
            max_tau,
            max_step=np.inf,
            rtol=1e-10,
            atol=1e-10,
            jac=None,
            jac_sparsity=jac_structure,
        )
        while True:
            sol.step()
            if sol.status in ("finished", "failed"):
                print("Equilibrium achieved/failed")
                break

            t_vals.append(sol.t)
            y_vals.append(sol.y.copy())
            if (
                max(
                    vector_of_magnitudes(
                        np.reshape(scipy_fun(None, sol.y), (np.shape(incidence_matrix)[1], 2))
                    )
                )
                < 1e-4
            ):
                print("Equilibrium achieved")
                break
            if sol.t >= 200:
                print("Equilibrium not achieved in 200 tau")

    ###############
    t_vals = [0] + t_vals
    energy_vals = [energy_calc(y)]
    norm_vals = [np.linalg.norm(scipy_fun(None, y))]
    for i in range(np.shape(y_vals)[0]):
        energy_vals.append(energy_calc(y_vals[i]))
        norm_vals.append(np.linalg.norm(scipy_fun(None, y_vals[i])))

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

    force_all = boundary_force(y_vals[-1])

    force_top = np.array([force_all[item] for item in top_nodes])

    force_bot = np.array([force_all[item] for item in bot_nodes])

    force_left = np.array([force_all[item] for item in left_nodes])

    force_right = np.array([force_all[item] for item in right_nodes])

    y_output = np.reshape(y_vals[-1], (np.shape(incidence_matrix)[1], 2))
    if (
        Plot_networks
        and np.linalg.norm(
            vector_of_magnitudes(incidence_matrix.dot(y_output)) / initial_lengths - 1
        )
        > 1e-12
    ):

        try:
            Fixed_BC_script.ColormapPlot_dilation(
                y_output,
                incidence_matrix,
                L,
                Lambda_1,
                Lambda_2,
                ((vector_of_magnitudes(incidence_matrix.dot(y_output)) / initial_lengths) - 1),
                r"$F_j$",
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
        [t_vals, norm_vals],
        y_output,  # The nodes in equilibrium positions
        energy_vals,
    )


#############################################################################
#############################################################################

# Currently does not properly deal with computing stress-strain results, needs abit of work


def Realisation_dilation(
    L,
    density,
    seed,
    fibre_lengths_multiplier,
    max_Lambda_1,
    max_Lambda_2,
    num_steps,
    Plot_stress_results=False,
    Plot_networks=False,
):
    realisation_start_time = time.time()
    data = []

    Lambda_1_step = (max_Lambda_1 - 1) / (num_steps - 1)
    Lambda_2_step = (max_Lambda_2 - 1) / (num_steps - 1)

    stresses = []

    (nodes, boundary_nodes, incidence_matrix) = Fixed_BC_script.Create_pbc_Network(
        L,
        density,
        seed,
    )

    initial_lengths = fibre_lengths_multiplier * vector_of_magnitudes(incidence_matrix.dot(nodes))

    total_time = time.time()

    input_nodes = np.copy(nodes)
    # flag_skipped_first_computation = 0
    for i in range(num_steps):
        # if i == 0 and fibre_lengths_multiplier == 1:
        #     flag_skipped_first_computation = 1
        #     continue
        data.append(
            Radau_timestepper_dilation(
                L,
                input_nodes,
                incidence_matrix,
                boundary_nodes,
                initial_lengths,
                1 + Lambda_1_step * i,
                1 + Lambda_2_step * i,
                Plot_networks,
            )
        )
        # Each step we provide a guess for the solution at the next step using the previous solution
        input_nodes = invert_dilation(
            data[i][-2],
            1 + Lambda_1_step * i,
            1 + Lambda_2_step * i,
        )

    print("Total Deformation Computational time = ", time.time() - total_time)

    (p_top, p_bot, p_left, p_right) = calculate_stress_strain_stretch(
        data, Lambda_1_step, Lambda_2_step, num_steps, L, fibre_lengths_multiplier
    )[-1]

    stresses.append([p_top, p_bot, p_left, p_right])

    if Plot_stress_results:
        Lambda_1 = np.linspace(1, max_Lambda_1, num_steps)
        Lambda_2 = np.linspace(1, max_Lambda_2, num_steps)
        if max_Lambda_1 == 1:
            Lambda_1 = Lambda_2
        if max_Lambda_2 == 1:
            Lambda_2 = Lambda_1

        plt.figure()

        plt.plot(Lambda_2, p_top[:, 0])
        plt.plot(Lambda_1, p_top[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Top boundary")
        plt.savefig(
            "Stress_strain_top_d{}_p{}_s{}_L1{}_L2{}.png".format(
                density, fibre_lengths_multiplier, seed, max_Lambda_1, max_Lambda_2
            )
        )

        plt.figure()

        plt.plot(Lambda_2, p_bot[:, 0])
        plt.plot(Lambda_1, p_bot[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Bottom boundary")
        plt.savefig(
            "Stress_strain_top_d{}_p{}_s{}_L1{}_L2{}.png".format(
                density, fibre_lengths_multiplier, seed, max_Lambda_1, max_Lambda_2
            )
        )

        plt.figure()

        plt.plot(Lambda_1, p_left[:, 0])
        plt.plot(Lambda_2, p_left[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Left boundary")
        plt.savefig(
            "Stress_strain_top_d{}_p{}_s{}_L1{}_L2{}.png".format(
                density, fibre_lengths_multiplier, seed, max_Lambda_1, max_Lambda_2
            )
        )

        plt.figure()

        plt.plot(Lambda_1, p_right[:, 0])
        plt.plot(Lambda_2, p_right[:, 1])
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend(["Shear Stress", "Normal Stress"])
        plt.title("Right boundary")
        plt.savefig(
            "Stress_strain_top_d{}_p{}_s{}_L1{}_L2{}.png".format(
                density, fibre_lengths_multiplier, seed, max_Lambda_1, max_Lambda_2
            )
        )
    with open(
        "Dilation_radau_realisation_{}_{}_{}_{}_{}_{}_{}_{}_{}_".format(
            L,
            density,
            seed,
            fibre_lengths_multiplier,
            max_Lambda_1,
            max_Lambda_2,
            num_steps,
            Plot_stress_results,
            Plot_networks,
        )
        + str(date.today())
        + ".dat",
        "wb",
    ) as f:
        pickle.dump(
            (data, [p_top, p_bot, p_left, p_right], (nodes, initial_lengths, incidence_matrix)), f
        )

    print("Total Realisation time =", time.time() - realisation_start_time)

    return (
        data,
        [p_top, p_bot, p_left, p_right],
        (nodes, initial_lengths, boundary_nodes, incidence_matrix),
    )


# Jacobian calculation is identical to hessian excepting the df_i/dr_k elements when f_i==0 as a
# consequence of boundary conditions.
def hessian_component(l_j_hat, stretch):
    l_j_outer = np.einsum("i,k", l_j_hat, l_j_hat)
    coefficient = 1 - 1 / stretch
    return l_j_outer - coefficient * (np.eye(2) - l_j_outer)


def jacobian(nodes, incidence_matrix, initial_lengths, boundary_nodes):
    num_edges, num_nodes = np.shape(incidence_matrix)
    l_j = incidence_matrix.dot(nodes)
    l_j_lengths = vector_of_magnitudes(l_j)
    l_j_hat = normalise_elements(l_j)
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
            stretch = l_j_lengths[edge] / initial_lengths[edge]
            component = -hessian_component(l_j_hat[edge], stretch)
            hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = component
            continue
        stretch = l_j_lengths[edge] / initial_lengths[edge]
        component = -hessian_component(l_j_hat[edge], stretch)
        hessian[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = component
        hessian[2 * k : 2 * k + 2, 2 * i : 2 * i + 2] = component

    # We now compute the more complex diagonal entries
    for node in range(boundary_nodes):
        edge_indices = np.nonzero(incidence_matrix[:, node])[0]
        i = 2 * node
        hessian[i : i + 2, i : i + 2] = sum(
            [
                hessian_component(l_j_hat[edge], l_j_lengths[edge] / initial_lengths[edge])
                for edge in edge_indices
            ]
        )
    return hessian


# Orientation angle is calculated as the angle in the range [0,\pi] that an edge makes with the
# positive x axis. This is calculated by taking the dot product of an edge vector with e_1 = [1,0]^T
# And rearanging the dot product formula (mod_pi) to obtain desired result


def Orientation_distribution(nodes, incidence_matrix, Plot_data=False):
    l_j = incidence_matrix.dot(nodes)
    lengths = vector_of_magnitudes(incidence_matrix.dot(nodes))
    orientations_output = []
    for i in range(len(l_j)):
        orientations_output.append(np.mod(np.arccos(l_j[i][0] / np.linalg.norm(l_j[i])), np.pi))
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
        plt.show()
    return orientations_output


def l_j_lengths_and_orientations_histogram_plot(
    nodes, incidence_matrix, bins_edges, bins_orientations, Lambda_1, Lambda_2
):
    orientations = Orientation_distribution(nodes, incidence_matrix)
    l_j_lengths = vector_of_magnitudes(incidence_matrix.dot(nodes))
    plt.figure()
    plt.hist(l_j_lengths, bins=bins_edges, density=True)
    plt.title(
        r"PDF of edge length distribution for $(\Lambda_1,\Lambda_2)$ = {}".format(
            Lambda_1, Lambda_2
        )
    )

    plt.figure()
    plt.hist(orientations, bins=bins_orientations, density=True)
    plt.title(
        r"PDF of fibre orientations for $(\Lambda_1,\Lambda_2)$ = {}".format(Lambda_1, Lambda_2)
    )
    return


def Stretch_distribution(nodes, incidence_matrix, initial_lengths, num_bins, Lambda_1, Lambda_2):
    stretches = []
    l_j = incidence_matrix.dot(nodes)
    l_j_lengths = vector_of_magnitudes(l_j)
    for i in range(len(l_j_lengths)):
        stretches.append(l_j_lengths[i] / initial_lengths[i])
    plt.figure()
    plt.hist(stretches, bins=num_bins, density=True)
    plt.title(r"PDF of fibre stretches for $(\Lambda_1,\Lambda_2)$ = {}".format(Lambda_1, Lambda_2))
    return


def affinity_strain_stretch(initial_nodes, input_nodes, incidence_matrix, Lambda_1, Lambda_2):
    initial_lengths = vector_of_magnitudes(incidence_matrix.dot(initial_nodes))
    affine_nodes = dilation_deformation(initial_nodes, Lambda_1, Lambda_2)
    L_affine = incidence_matrix.dot(affine_nodes)
    L_affine_lengths = vector_of_magnitudes(L_affine)
    l_j = incidence_matrix.dot(input_nodes)
    l_j_lengths = vector_of_magnitudes(l_j)
    strain_affine = (L_affine_lengths - initial_lengths) / initial_lengths
    strain_actual = (l_j_lengths - initial_lengths) / initial_lengths
    return np.linalg.norm(strain_actual - strain_affine) / (
        len(initial_lengths) * np.linalg.norm(strain_affine)
    )


def affine_stretch_calc(orientation, Lambda_1, Lambda_2):
    return np.sqrt(
        Lambda_1**2 * np.cos(orientation) ** 2 + Lambda_2**2 * np.sin(orientation) ** 2
    )


def affine_stretch_def_calc(orientation, Lambda_1, Lambda_2):
    return np.sqrt(
        (Lambda_1**2 * Lambda_2**2)
        / (Lambda_2**2 * np.cos(orientation) ** 2 + Lambda_1**2 * np.sin(orientation) ** 2)
    )


def lambda_p_undeformed(stretches, orientations, Lambda_1, Lambda_2):
    output = []
    for i, item in enumerate(orientations):
        output.append(stretches[i] / affine_stretch_calc(item, Lambda_1, Lambda_2))
    return output


def lambda_p_prestressed(stretches, orientations, Lambda_1, Lambda_2, initial_stretches):
    return lambda_p_undeformed(stretches, orientations, Lambda_1, Lambda_2) * initial_stretches


def lambda_p_deformed(stretches, orientations, Lambda_1, Lambda_2):
    output = []
    for i, item in enumerate(orientations):
        output.append(stretches[i] / affine_stretch_def_calc(item, Lambda_1, Lambda_2))
    return output


def affine_theta_calc(Lambda_1, Lambda_2, initial_thetas):
    return [
        np.arcsin(
            Lambda_2
            * np.sin(item)
            / np.sqrt(Lambda_1**2 * np.cos(item) ** 2 + Lambda_2**2 * np.sin(item) ** 2)
        )
        if 0 <= item <= np.pi / 2
        else np.pi
        - np.arcsin(
            Lambda_2
            * np.sin(item)
            / np.sqrt(Lambda_1**2 * np.cos(item) ** 2 + Lambda_2**2 * np.sin(item) ** 2)
        )
        for item in initial_thetas
    ]


def stretch_prediction_gamma(k, theta, Lambda_1, Lambda_2, min_stretch, max_stretch):
    lambda_inputs = np.linspace(min_stretch - 0.05, max_stretch, 10000)
    output = []
    alpha = k - 1
    e = np.exp(1)
    for L in lambda_inputs:
        integrand = lambda x: (2 / np.pi) * (
            e ** (alpha * np.log((L * e) / (x * alpha * theta)) - L / (x * theta))
            / (
                np.sqrt(2 * np.pi * alpha * theta**2)
                * np.sqrt((Lambda_1**2 - x**2) * (x**2 - Lambda_2**2))
            )
        )
        output.append(
            sp.integrate.quadrature(integrand, Lambda_2, Lambda_1, tol=1.49e-08, rtol=1.49e-08)[0]
        )
    return np.array(output)


def stretch_prediction_lognorm(mu, sigma, Lambda_1, Lambda_2, min_stretch, max_stretch):
    lambda_inputs = np.linspace(min_stretch - 0.05, max_stretch, 10000)
    output = []
    for L in lambda_inputs:
        integrand = lambda x: (2 / np.pi) * (
            x
            * np.exp(-((np.log(L / x) - mu) ** 2) / (2 * sigma**2))
            / (
                L
                * sigma
                * np.sqrt(2 * np.pi)
                * np.sqrt((Lambda_1**2 - x**2) * (x**2 - Lambda_2**2))
            )
        )
        output.append(
            sp.integrate.quadrature(integrand, Lambda_2, Lambda_1, tol=1.49e-08, rtol=1.49e-08)[0]
        )
    return output


def stretch_prediction_st(shape_st, loc_st, scale_st, Lambda_1, Lambda_2, min_stretch, max_stretch):

    lambda_inputs = np.linspace(min_stretch - 0.05, max_stretch, 10000)
    output = []
    for L in lambda_inputs:
        integrand = lambda x: (
            2
            * sp.special.gamma((shape_st + 1) / 2)
            / (sp.special.gamma((shape_st) / 2) * np.pi * scale_st)
        ) * (
            [1 + ((L / x - loc_st) / scale_st) ** 2 / shape_st] ** (-(shape_st + 1) / 2)
            / (
                np.sqrt(np.pi * shape_st)
                * np.sqrt((Lambda_1**2 - x**2) * (x**2 - Lambda_2**2))
            )
        )

        output.append(
            sp.integrate.quadrature(integrand, Lambda_2, Lambda_1, tol=1.49e-08, rtol=1.49e-08)[0]
        )
    return output


def Spline_derivative_plot(input_data, strains, knots, derivative_order, data_name):
    input_spl = sp.interpolate.UnivariateSpline(strains, input_data, k=knots)
    plt.figure()
    plt.plot(strains, input_data, "ro")
    plt.plot(strains, input_spl(strains))
    plt.xlabel(r"$\Lambda_2$")
    plt.ylabel(data_name)
    # plt.savefig("sigma_yy_p09_biaxial.pdf")

    plt.show()
    input_spl_deriv = input_spl.derivative(n=derivative_order)
    plt.figure()
    derivs = input_spl_deriv(strains)
    plt.plot(strains, derivs)
    plt.plot(strains, derivs, "ko")
    plt.xlabel(r"$\Lambda_2$")
    plt.ylabel("Order {} derivative of {}".format(derivative_order, data_name))
    # plt.savefig("deriv_sigma_yy_p09_biaxial.pdf")
    return


def discrete_stress(nodes, incidence_matrix, initial_lengths, L):
    l_j = incidence_matrix.dot(nodes)

    normalised_edges = normalise_elements(l_j)

    l_j = vector_of_magnitudes(l_j)

    F_j = (l_j - initial_lengths) / initial_lengths

    sigma = np.matrix([[0, 0], [0, 0]])

    for index in range(len(initial_lengths)):
        sigma = sigma + F_j[index] * l_j[index] * np.outer(
            normalised_edges[index], normalised_edges[index]
        )

    return sigma / (L**2)


# Profiling Stuff


# def run_once():
#     # >>> put in the actual arguments you use <<<
#     L = 2
#     density = 10
#     seed = 4

#     Lambda_1 = 2.8
#     Lambda_2 = 2.8

#     (nodes, boundary_nodes, incidence_matrix) = Fixed_BC_script.Create_pbc_Network(
#         L,
#         density,
#         seed,
#     )

#     initial_lengths = vector_of_magnitudes(incidence_matrix.dot(nodes))

#     Radau_timestepper_dilation(
#         L,
#         nodes,
#         incidence_matrix,
#         boundary_nodes,
#         initial_lengths,
#         Lambda_1,
#         Lambda_2,
#         Plot_networks=False,  # avoid profiling plotting
#     )


# if __name__ == "__main__":
#     run_once()
