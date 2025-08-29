######################################
from Dilation_radau import *  # noqa
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import scipy as sp
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar

file_path = os.path.realpath(__file__)
sys.path.append(file_path)
import Fixed_BC_script  # noqa


#############################################################################
# Processing data to get lambda_p data

import re
import os
import pickle
import numpy as np
import scipy as sp

from Dilation_radau import *

print([sys.argv[1], sys.argv[2], sys.argv[3]])

# Updated regex pattern to match filenames
pattern = r"Dilation_radau_realisation_(\d+)_(\d+)\.0_(\d+)_.*\.dat"

# Directory containing your .dat files
directory = sys.argv[1]

p = float(sys.argv[2])

flag_first_or_second_deformation = int(sys.argv[3])

if p != 1:
    stretch = (
        1 - flag_first_or_second_deformation
    ) * 1 / p + flag_first_or_second_deformation * 1.2 / p

# Prepare final results

rho_dict = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 16: 6, 20: 7, 25: 8, 30: 9}

lambda_p_converged = [[[] for _ in range(10)] for _ in range(10)]

for filename in os.listdir(directory):
    match = re.match(pattern, filename)
    if match:
        X, Y, Z = map(int, match.groups())
        x_index = X - 1
        y_index = rho_dict[Y]

        file_path = os.path.join(directory, filename)

        # Load and process immediately
        with open(file_path, "rb") as f:
            item = pickle.load(f)
        nodes = item[0][int(sys.argv[3])][-2] / p
        initial_nodes = np.array(item[-1][0]) / p
        incidence_matrix = item[-1][-1]
        initial_lengths = item[-1][1] / p

        L = X
        for k, node in enumerate(initial_nodes[::-1]):
            if not (
                any([abs(item - 0) <= 1e-15 for item in node])
                or any([abs(item - L) <= 1e-15 for item in node])
            ):
                boundary_nodes = len(nodes) - k

        def scipy_fun(t, y):
            matrix_y = np.reshape(y, (np.shape(incidence_matrix)[1], 2))
            l_j = incidence_matrix.dot(matrix_y)
            l_j_hat = normalise_elements(l_j)
            F_j = (np.sqrt(np.einsum("ij,ij->i", l_j, l_j)) - initial_lengths) / initial_lengths
            product = np.einsum("ij,i->ij", l_j_hat, F_j)
            f_jk = incidence_matrix.T.dot(product)
            f_jk[boundary_nodes:] = 0
            return -np.reshape(f_jk, 2 * np.shape(f_jk)[0], order="C")

        if not (
            max(
                vector_of_magnitudes(
                    np.reshape(scipy_fun(None, nodes), (np.shape(incidence_matrix)[1], 2))
                )
            )
            < 1e-04
        ):
            continue
        stretches = vector_of_magnitudes(incidence_matrix.dot(nodes)) / initial_lengths
        theta = Orientation_distribution(initial_nodes, incidence_matrix, False)
        lambda_p = lambda_p_undeformed(stretches, theta, stretch, stretch)

        # Store only final results

        lambda_p_converged[x_index][y_index].append(lambda_p)
with open(
    "lambda_p_{}.dat".format(stretch),
    "wb",
) as f:
    pickle.dump(lambda_p_converged, f)

#############################################################################
# Computing sample values of df, mean, variance, and scale


def fit_t_with_fixed_moments(data):
    mean = np.mean(data)
    var = np.var(data, ddof=1)  # Unbiased sample variance

    def neg_log_likelihood(df):
        if df <= 2:
            return RuntimeError("DoF must be > 2")  # Variance undefined for df ≤ 2

        scale = np.sqrt(var * (df - 2) / df)
        return -np.sum(sp.stats.t.logpdf(data, df, loc=mean, scale=scale))

    result = minimize_scalar(neg_log_likelihood, bounds=(2.01, 1000), method="bounded")

    if result.success:
        df_est = result.x
        scale_est = np.sqrt(var * (df_est - 2) / df_est)
        return df_est, mean, scale_est, var
    else:
        raise RuntimeError("fitting failed")


df = [[[] for _ in range(len(lambda_p_converged[i]))] for i in range(10)]
mu = [[[] for _ in range(len(lambda_p_converged[i]))] for i in range(10)]
var = [[[] for _ in range(len(lambda_p_converged[i]))] for i in range(10)]
tau = [[[] for _ in range(len(lambda_p_converged[i]))] for i in range(10)]

for i, L in enumerate(lambda_p_converged):
    for j, rho in enumerate(L):
        for seed in rho:
            df_t, mu_t, scale_t, var_t = fit_t_with_fixed_moments(seed)

            df[i][j].append(df_t)
            mu[i][j].append(mu_t)
            tau[i][j].append(scale_t)
            var[i][j].append(var_t)

mu_means = [[np.mean(mu[i][j]) for j in range(len(df[i]))] for i in range(10)]
mu_std = [[np.std(mu[i][j]) for j in range(len(df[i]))] for i in range(10)]
mu_bxp = [
    [
        (
            min(mu[i][j]),
            np.quantile(mu[i][j], 0.25),
            np.quantile(mu[i][j], 0.5),
            np.quantile(mu[i][j], 0.75),
            max(mu[i][j]),
        )
        for j in range(len(df[i]))
    ]
    for i in range(10)
]

var_means = [[np.mean(var[i][j]) for j in range(len(df[i]))] for i in range(10)]
var_std = [[np.std(var[i][j]) for j in range(len(df[i]))] for i in range(10)]
var_bxp = [
    [
        (
            min(var[i][j]),
            np.quantile(var[i][j], 0.25),
            np.quantile(var[i][j], 0.5),
            np.quantile(var[i][j], 0.75),
            max(var[i][j]),
        )
        for j in range(len(df[i]))
    ]
    for i in range(10)
]

tau_means = [[np.mean(tau[i][j]) for j in range(len(df[i]))] for i in range(10)]
tau_std = [[np.std(tau[i][j]) for j in range(len(df[i]))] for i in range(10)]
tau_bxp = [
    [
        (
            min(tau[i][j]),
            np.quantile(tau[i][j], 0.25),
            np.quantile(tau[i][j], 0.5),
            np.quantile(tau[i][j], 0.75),
            max(tau[i][j]),
        )
        for j in range(len(df[i]))
    ]
    for i in range(10)
]

df_means = [[np.mean(df[i][j]) for j in range(len(df[i]))] for i in range(10)]
df_std = [[np.std(df[i][j]) for j in range(len(df[i]))] for i in range(10)]
df_bxp = [
    [
        (
            min(df[i][j]),
            np.quantile(df[i][j], 0.25),
            np.quantile(df[i][j], 0.5),
            np.quantile(df[i][j], 0.75),
            max(df[i][j]),
        )
        for j in range(len(df[i]))
    ]
    for i in range(10)
]

#############################################################################
# Mu curve fitting and plots


def model_mu(x, a, m):
    return a * x ** (-m) + 1


rho_vals = np.array([4, 6, 8, 10])

a_fits_mu = []
m_fits_mu = []
b_fits_mu = []

L = 1
for j, item in enumerate(mu_means):
    y = np.array(item[1:])

    try:
        params, covariance = curve_fit(model_mu, rho_vals, y, p0=[-1, 1], method="trf")
        a_fit, m_fit = params
        a_fits_mu.append(a_fit)
        m_fits_mu.append(m_fit)
        print(f"μ fits done for L = {L}")
    except Exception as e:
        print(f"μ fits FAILED for L = {L}: {e}")
    L += 1

for i in range(len(a_fits_mu)):
    print(f"L={i+1}: a={a_fits_mu[i]:.4f}, m={m_fits_mu[i]:.4f}")


#############################################################################
# Var curve fitting and plots


def model_var(x, a, K):
    return np.log(a) - K * np.log(x)


rho_vals = np.array([4, 6, 8, 10])

a_fits_var = []
K_fits_var = []
S_fits_var = []

L = 1
for j, item in enumerate(var_means):
    y = np.array(item[1:])

    try:
        params, covariance = curve_fit(model_var, rho_vals, np.log(y), p0=[1, 1], method="trf")
        a_fit, K_fit = params
        a_fits_var.append(a_fit)
        K_fits_var.append(K_fit)
        print(f"σ² fits done for L = {L}")
    except Exception as e:
        print(f"σ² fits FAILED for L = {L}: {e}")
    L += 1

for i in range(len(a_fits_var)):
    print(f"L={i+1}: a={a_fits_var[i]:.4f}, K={K_fits_var[i]:.4f}")


#############################################################################
# Tau curve fitting and plots


def model_tau(x, a, T):
    return np.log(a) - T * np.log(x)


rho_vals = np.array([4, 6, 8, 10])

a_fits_tau = []
T_fits_tau = []
D_fits_tau = []

L = 1
for j, item in enumerate(tau_means):
    y = np.array(item[1:])

    try:
        params, covariance = curve_fit(model_tau, rho_vals, np.log(y), p0=[1, 1], method="trf")
        a_fit, T_fit = params
        a_fits_tau.append(a_fit)
        T_fits_tau.append(T_fit)
        print(f"τ fits done for L = {L}")
    except Exception as e:
        print(f"τ fits FAILED for L = {L}: {e}")
    L += 1

for i in range(len(a_fits_tau)):
    print(f"L={i+1}: a={a_fits_tau[i]:.4f}, T={T_fits_tau[i]:.4f}")


#############################################################################
with open("results_mu_var_tau_df_store_{}.dat".format(stretch), "wb") as f:
    pickle.dump(
        (
            (mu_means, mu_bxp, mu_std),
            (var_means, var_bxp, var_std),
            (tau_means, tau_bxp, tau_std),
            (df_means, df_bxp, df_std),
        ),
        f,
    )
