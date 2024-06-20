# For running code on the maths cs servers, will figure out how to do shit properly later

######################################
import os
import sys
import pickle

file_path = os.path.realpath(__file__)
sys.path.append(file_path)
import Biaxial_stretch_Radau  # noqa
import Stretch_Radau  # noqa
import PBC_network  # noqa
import Dilation_radau  # noqa
import Make_plots  # noqa


######################################
def test():
    # if len(sys.argv) > 1:
    #     L = sys.argv[1]
    # else:
    #     L = 2
    L = 2
    fibre_lengths_multiplier = 1
    num_intervals = 16

    new_data = []
    network_data = []
    energy_data = []

    with open("readme.txt", "w") as read:
        read.write("Info on how far code has progressed")
    for density_loop in range(1):
        density = 40  # (3 + density_loop) * 10

        with open("readme.txt", "a") as read:
            read.write("\n")
            read.write("Density = ".format(density))

        new_data.append([])
        network_data.append([])
        energy_data.append([])
        for seed in range(1):
            loop_data = Stretch_Radau.Single_realisation_uniaxial_stretch(
                L, density, 2, 1, 0.2, num_intervals, False, False
            )

            Network = PBC_network.CreateNetwork(density * L**2, L, 5)

            pbc_edges = Network[0][2]
            pbc_nodes = Network[1][2]
            pbc_incidence_matrix = Network[2][1]

            # fig_network = plt.figure()
            # PBC_network.PlotNetwork(pbc_edges, L)
            # plt.savefig("Network_L3_seed2_N81.pdf")

            (
                nodes,
                edges,
                incidence_matrix,
                boundary_nodes,
                top_nodes,
                bot_nodes,
                left_nodes,
                right_nodes,
            ) = Biaxial_stretch_Radau.restructure_PBC_data(
                pbc_edges, pbc_nodes, pbc_incidence_matrix, L
            )

            initial_lengths = fibre_lengths_multiplier * Biaxial_stretch_Radau.vector_of_magnitudes(
                incidence_matrix.dot(nodes)
            )

            new_data[density_loop].append(loop_data)
            network_data[density_loop].append([nodes, initial_lengths, incidence_matrix])
            energy_data[density_loop].append([])
            for energy_loop in range(num_intervals):
                energy_data[density_loop][seed].append(loop_data[0][energy_loop][-1][-1])

            with open("output_data_inloop.dat", "wb") as loop_info:
                pickle.dump((new_data, network_data, energy_data), loop_info)

        print("Density LOOP COMPLETED")
    return (new_data, network_data, energy_data)


L = 2

(
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
) = Make_plots.produce_data(2, 10, 1, 0, [1.2, 2], [1, 1])


output_data = (
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
)


with open("Test_data.dat", "wb") as f:
    pickle.dump(output_data, f)
