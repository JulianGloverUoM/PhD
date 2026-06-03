# Testing script

######################################
from dataclasses import dataclass
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


import time
import itertools
import numpy as np
import pandas as pd

######################################


@dataclass(frozen=True)
class Fibre_Law:
    name: str
    force: callable  # F(lambda)
    energy: callable  # U(lambda, L0)
    stiffness: callable  # dF(lambda)/dlambda


def make_law(key, alpha=None):
    if key == "Hookean_spring":
        return Fibre_Law(
            name="Hookean_spring",
            force=lambda stretch: stretch - 1,
            energy=lambda stretch, L_0: 0.5 * L_0 * (stretch - 1) ** 2,
            stiffness=lambda stretch: 1,
        )

    elif key == "Neo_Hookean":  # example placeholder
        return Fibre_Law(
            name="Neo_Hookean",
            force=lambda stretch: (1 / 3) * (stretch - stretch**-2),
            energy=lambda stretch, L_0: (L_0 / 6) * (stretch**2 + 2 / stretch - 3),
            stiffness=lambda stretch: (1 / 3) * (1 + 2 * stretch**-3),
        )
    elif key == "Soft_compression":
        if alpha is None:
            raise ValueError("Soft_compression requires alpha.")
        if not (0.0 <= alpha < 1.0):
            raise ValueError("Soft_compression requires 0 <= alpha < 1.")

        return Fibre_Law(
            name=f"Soft_compression_alpha_{alpha}",
            force=lambda stretch: np.where(
                stretch >= 1.0,
                stretch - 1.0,
                alpha * (stretch - 1.0),
            ),
            energy=lambda stretch, L_0: 0.5
            * L_0
            * np.where(
                stretch >= 1.0,
                (stretch - 1.0) ** 2,
                alpha * (stretch - 1.0) ** 2,
            ),
            stiffness=lambda stretch: np.where(
                stretch >= 1.0,
                1,
                alpha,
            ),
        )
    elif key == "Log_law":  # example placeholder
        return Fibre_Law(
            name="Log_law",
            force=lambda stretch: np.log(stretch),
            energy=lambda stretch, L_0: L_0 * stretch * (np.log(stretch) - 1),
            stiffness=lambda stretch: 1 / stretch,
        )
    else:
        raise ValueError(f"Unknown fibre law: {key}")


######################################


def dilation_deformation(input_nodes, Lambda_1, Lambda_2):
    return input_nodes * np.array([Lambda_1, Lambda_2])


def normalise_elements(A):
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / norms


def vector_of_magnitudes(A):
    return np.linalg.norm(A, axis=1)


def frobenius_norm(A):
    return np.linalg.norm(A)


######################################


def timestepper(
    L,
    nodes,
    incidence_matrix,
    boundary_nodes,
    initial_lengths,
    Lambda_1,
    Lambda_2,
    solver="BDF",
    Plot_networks=False,
    Fibre_law="Hookean_spring",
    alpha=0.1,
):
    if solver == "Radau":
        SolverClass = sp.integrate.Radau
    elif solver == "BDF":
        SolverClass = sp.integrate.BDF
    else:
        raise ValueError(f"Unknown solver '{solver}', use 'Radau' or 'BDF'")
    law = make_law(Fibre_law, alpha)
    (num_edges, num_nodes) = incidence_matrix.shape
    inv_initial_lengths = 1.0 / initial_lengths

    def scipy_fun(t, y):
        matrix_y = y.reshape(num_nodes, 2)
        l_j = incidence_matrix.dot(matrix_y)
        l_j_lengths = np.linalg.norm(l_j, axis=1)
        l_j_hat = l_j / l_j_lengths[:, None]
        stretches = l_j_lengths * inv_initial_lengths
        F_j = law.force(stretches)
        product = l_j_hat * F_j[:, None]
        f_jk = incidence_matrix.T.dot(product)
        f_jk[boundary_nodes:] = 0
        return -f_jk.ravel(order="C")

    def boundary_force(y):
        matrix_y = y.reshape(num_nodes, 2)
        l_j = incidence_matrix.dot(matrix_y)
        l_j_lengths = np.linalg.norm(l_j, axis=1)
        l_j_hat = l_j / l_j_lengths[:, None]
        stretches = l_j_lengths * inv_initial_lengths
        F_j = law.force(stretches)
        product = l_j_hat * F_j[:, None]
        f_jk = incidence_matrix.T.dot(product)
        return -f_jk

    def energy_calc(y):
        matrix_y = y.reshape(num_nodes, 2)
        l_j = incidence_matrix.dot(matrix_y)
        l_j_lengths = np.linalg.norm(l_j, axis=1)
        stretches = l_j_lengths * inv_initial_lengths
        return np.sum(law.energy(stretches, initial_lengths))

    def jacobian(t, y):
        matrix_y = y.reshape(num_nodes, 2)

        jac = lil_matrix((2 * num_nodes, 2 * num_nodes))

        l_j = incidence_matrix.dot(matrix_y)
        l_j_lengths = np.linalg.norm(l_j, axis=1)
        l_j_hat = l_j / l_j_lengths[:, None]

        l_j_hat_outer_products = l_j_hat[:, :, None] * l_j_hat[:, None, :]

        stretches = l_j_lengths * inv_initial_lengths

        F_j = law.force(stretches)
        k_j = law.stiffness(stretches)

        I_2 = np.eye(2)

        hessian_components = -(
            (F_j / l_j_lengths)[:, None, None] * I_2
            + (k_j / initial_lengths - F_j / l_j_lengths)[:, None, None] * l_j_hat_outer_products
        )

        # This first loop computes the upper triangular off-diagonal blocks of the Jacobian.
        for edge in range(num_edges):
            i, k = incidence_matrix.getrow(edge).indices
            component = hessian_components[edge]
            # These checks incorporate the Neumann BCs. Note as we are calculating just the upper triangular block, i<k
            # Hence if i is on the boundary k must be as well, if i is not a boundary node then k might be, in which
            # case df_i/dr_k = 0 but df_k/dr_i =/= 0 so we calculate it.
            if i >= boundary_nodes:
                continue

            if k >= boundary_nodes:
                jac[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = component
                continue

            jac[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = component
            jac[2 * k : 2 * k + 2, 2 * i : 2 * i + 2] = component

        # We now compute the more complex diagonal entries
        for node in range(boundary_nodes):
            edge_indices = incidence_matrix[:, node].nonzero()[0]
            i = 2 * node

            jac[i : i + 2, i : i + 2] = -sum(hessian_components[edge] for edge in edge_indices)
        return jac.tocsr()

    # Effectively computes the Jacobian but with all potential non-zero elements = 1, which
    # gives the sparsity structure of the Jacobian without having to do the expensive computation
    def jac_sparsity_structure(y):
        num_edges, num_nodes = np.shape(incidence_matrix)
        jac = lil_matrix((2 * num_nodes, 2 * num_nodes))
        component = np.ones((2, 2))
        # if two edges are incident and one isnt a boundary node, make its local component's non-zero
        for edge in range(num_edges):
            i, k = incidence_matrix.getrow(edge).indices
            # These checks incorporate the Neumann BCs. Note as we are calculating just the upper triangular block, i<k
            # Hence if i is on the boundary k must be as well, if i is not a boundary node then k might be, in which
            # case df_i/dr_k = 0 but df_k/dr_i =/= 0 so we calculate it.
            if i >= boundary_nodes:
                continue

            if k >= boundary_nodes:
                jac[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = component
                continue

            jac[2 * i : 2 * i + 2, 2 * k : 2 * k + 2] = component
            jac[2 * k : 2 * k + 2, 2 * i : 2 * i + 2] = component
        # Make all diagonal entries non-zero, except the boundary nodes
        for node in range(boundary_nodes):
            i = 2 * node
            jac[i : i + 2, i : i + 2] = component
        return jac.tocsr()

    # start_time = time.time()

    y = dilation_deformation(nodes, Lambda_1, Lambda_2)
    y = np.reshape(y, 2 * np.shape(y)[0], order="C")

    jac_structure = jac_sparsity_structure(y)

    ###############
    # Timestepping

    increasing_energy = False
    slow_convergence = False

    max_tau = 500.0

    sol = SolverClass(
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

    t_val = 0.0
    y_val = y.copy()

    t_old = None
    y_old = None

    max_force = np.inf

    energy_val = energy_calc(y_val)
    energy_vals = [energy_val]
    norm_vals = [np.linalg.norm(scipy_fun(None, y_val))]
    t_vals = [t_val]

    while True:
        sol.step()

        if sol.status in ("finished", "failed"):
            print("Equilibrium failed")
            break

        new_t = sol.t
        new_y = sol.y.copy()
        new_energy = energy_calc(new_y)

        if y_old is not None and new_energy > energy_val:
            print("Energy increasing")
            increasing_energy = True
            break

        rhs = scipy_fun(None, new_y)
        force_magnitudes = vector_of_magnitudes(rhs.reshape(num_nodes, 2))
        max_force = np.max(force_magnitudes)

        t_old = t_val
        y_old = y_val

        t_val = new_t
        y_val = new_y
        energy_val = new_energy

        t_vals.append(t_val)
        energy_vals.append(energy_val)
        norm_vals.append(np.linalg.norm(rhs))

        if max_force < 1e-4:
            print("Equilibrium achieved")
            break

        if sol.t >= 100:
            print("Slow convergence")
            slow_convergence = True
            print(max_force)
            break

    if increasing_energy or slow_convergence:
        hundereds_count = 0

        sol = SolverClass(
            scipy_fun,
            t_val,
            y_val,
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
                print("Equilibrium failed")
                break

            new_t = sol.t
            new_y = sol.y.copy()

            rhs = scipy_fun(None, new_y)
            force_magnitudes = vector_of_magnitudes(rhs.reshape(num_nodes, 2))
            max_force = np.max(force_magnitudes)

            t_old = t_val
            y_old = y_val

            t_val = new_t
            y_val = new_y
            energy_val = energy_calc(y_val)

            t_vals.append(t_val)
            energy_vals.append(energy_val)
            norm_vals.append(np.linalg.norm(rhs))

            if max_force < 1e-4:
                print("Equilibrium achieved")
                break

            if sol.t >= 100 * (2 + hundereds_count):
                print("Equilibrium not achieved in {} tau".format(100 * (2 + hundereds_count)))
                hundereds_count += 1

    y_output = y_val.reshape(num_nodes, 2)
    if Plot_networks and (max_force < 1e-4):
        try:
            stretches = vector_of_magnitudes(incidence_matrix @ y_output) / initial_lengths
            Fixed_BC_script.ColormapPlot_dilation(
                y_output,
                incidence_matrix,
                L,
                Lambda_1,
                Lambda_2,
                stretches,
                r"$\lambda_j$",
            )
        except (IndexError, ZeroDivisionError, ValueError):
            pass
    return (
        [t_vals, norm_vals],
        y_output,
        energy_vals,
    )


def benchmark_solvers(
    L_values,
    density_values,
    seeds,
    Lambda_1,
    Lambda_2,
    rtol=1e-6,
    atol=1e-6,
):
    results = []

    for L, density, seed in itertools.product(L_values, density_values, seeds):

        (nodes, boundary_nodes, incidence_matrix) = Fixed_BC_script.Create_pbc_Network(
            L,
            density,
            seed,
        )

        initial_lengths = vector_of_magnitudes(incidence_matrix.dot(nodes))
        for solver in ("Radau", "BDF"):
            t0 = time.perf_counter()
            try:
                ([t_vals, norm_vals], y_output, energy_vals,) = timestepper(
                    L,
                    nodes,
                    incidence_matrix,
                    boundary_nodes,
                    initial_lengths,
                    Lambda_1,
                    Lambda_2,
                    solver=solver,
                    Plot_networks=False,
                )
                status = "ok"
                print(L, density, seed, solver)
            except Exception as e:
                # in case one solver fails for some parameter set
                status = f"error: {type(e).__name__}"
                print(L, density, seed, solver)
            t1 = time.perf_counter()

            results.append(
                {
                    "solver": solver,
                    "L": L,
                    "density": density,
                    "seed": seed,
                    "rtol": rtol,
                    "atol": atol,
                    "time_s": t1 - t0,
                    "status": status,
                }
            )

    # Optional: convert to DataFrame for easy sorting/plotting in Spyder
    try:
        df = pd.DataFrame(results)
    except Exception:
        return results

    # Sort for readability
    df = df.sort_values(["L", "density", "seed", "solver"]).reset_index(drop=True)
    return df


def run_once():
    L_values = [
        2,
        4,
    ]
    density_values = [4]
    seeds = [0, 1, 2]

    df = benchmark_solvers(L_values, density_values, seeds, Lambda_1=1.2, Lambda_2=1.2)
    print(df)
    # with open("profiling_output.dat", "wb") as f:
    #     pickle.dump(df, f)


if __name__ == "__main__":
    run_once()
