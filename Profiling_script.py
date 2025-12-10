# Testing script

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


import time
import itertools
import numpy as np
import pandas as pd

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
):
    if solver == "Radau":
        SolverClass = sp.integrate.Radau
    elif solver == "BDF":
        SolverClass = sp.integrate.BDF
    else:
        raise ValueError(f"Unknown solver '{solver}', use 'Radau' or 'BDF'")

    num_nodes = incidence_matrix.shape[1]
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

    def energy_calc(y):
        matrix_y = np.reshape(y, (np.shape(incidence_matrix)[1], 2))
        l_j = incidence_matrix.dot(matrix_y)
        u_j = vector_of_magnitudes(l_j) - initial_lengths
        return 0.5 * np.matmul(1 / initial_lengths, np.square(u_j))

    def jac_sparsity_structure(y):
        num_edges, num_nodes = np.shape(incidence_matrix)
        hessian = lil_matrix((2 * num_nodes, 2 * num_nodes))
        component = np.ones((2, 2))

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

    y = dilation_deformation(nodes, Lambda_1, Lambda_2)
    y = np.reshape(y, 2 * np.shape(y)[0], order="C")

    y_vals = []
    t_vals = []

    jac_structure = jac_sparsity_structure(y)

    ###############
    # Timestepping

    increasing_energy = False
    slow_convergence = False

    max_tau = 1000.0

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

    t_vals = [0.0]
    y_vals = [y.copy()]

    while True:
        sol.step()
        if sol.status in ("finished", "failed"):
            print("Equilibrium failed")
            break

        if len(y_vals) > 2 and energy_calc(sol.y) > energy_calc(y_vals[-2]):
            print("Energy increasing")
            increasing_energy = True
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

        # equilibrium
        if sol.t >= 100:
            print("Slow convergence")
            slow_convergence = True
            print(
                max(
                    vector_of_magnitudes(
                        np.reshape(scipy_fun(None, sol.y), (np.shape(incidence_matrix)[1], 2))
                    )
                )
            )
            break

    # Some networks contain edges or structures that are stiff and require stricter error
    # tolerances for the RK23 scheme to converge to a mechanical equilibrium.
    # However these stricter error tolerances also increase the computational cost
    # of the scheme to we implement a test to only increase the tolerance when required.
    if increasing_energy or slow_convergence:
        hundereds_count = 0
        sol = SolverClass(
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
                print("Equilibrium failed")
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
            if sol.t >= 100 * (2 + hundereds_count):
                print("Equilibrium not achieved in {} tau".format(100 * (2 + hundereds_count)))
                hundereds_count += 1
                print(
                    max(
                        vector_of_magnitudes(
                            np.reshape(scipy_fun(None, sol.y), (np.shape(incidence_matrix)[1], 2))
                        )
                    )
                )

    ###############
    t_vals = [0] + t_vals
    energy_vals = [energy_calc(y)] + [energy_calc(item) for item in y_vals]
    norm_vals = [np.linalg.norm(scipy_fun(None, y))] + [
        np.linalg.norm(scipy_fun(None, item)) for item in y_vals
    ]

    y_output = np.reshape(y_vals[-1], (np.shape(incidence_matrix)[1], 2))
    if Plot_networks and (
        max(
            vector_of_magnitudes(
                np.reshape(scipy_fun(None, y_output), (np.shape(incidence_matrix)[1], 2))
            )
        )
        < 1e-4
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
    L_values = [2, 3, 4]
    density_values = [4, 6]
    seeds = [0, 1, 2]

    df = benchmark_solvers(L_values, density_values, seeds, Lambda_1=1.2, Lambda_2=1.2)
    print(df)
    with open("profling_output.dat", "wb") as f:
        pickle.dump(df, f)


if __name__ == "__main__":
    run_once()
