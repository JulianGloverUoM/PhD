# For running code on the maths cs servers, will figure out how to do shit properly later

######################################
import os
import sys
import pickle

file_path = os.path.realpath(__file__)
sys.path.append(file_path)
import Biaxial_stretch_Radau  # noqa
import PBC_network  # noqa

######################################
def test():
    # if len(sys.argv) > 1:
    #     L = sys.argv[1]
    # else:
    #     L = 2
    L = 2
    fibre_lengths_multiplier = 1

    new_data = []
    network_data = []
    energy_data = []

    with open("readme.txt", "w") as read:
        read.write("Info on how far code has progressed")
    for density_loop in range(5):
        density = (3 + density_loop) * 10

        with open("readme.txt", "a") as read:
            read.write("\n")
            read.write("Density = ".format(density))

        new_data.append([])
        network_data.append([])
        energy_data.append([])
        for seed in range(5):
            loop_data = Biaxial_stretch_Radau.Single_realisation_stretch(
                L, density, seed, 1, 0.5, 2, False, False
            )

            Network = PBC_network.CreateNetwork(density * L**2, L, seed)

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
            energy_data[density_loop].append(loop_data[0][0][-1][-1])

            with open("output_data_inloop.dat", "wb") as loop_info:
                pickle.dump((new_data, network_data, energy_data), loop_info)

        print("Density LOOP COMPLETED")
    return (new_data, network_data, energy_data)


# output_data = test()


# with open("output_data.dat", "wb") as f:
#     pickle.dump(test(), f)
