# Script to generate a PBC network with data structure in compliance with rerquirements for solving
# for equilibrium positions using dispersive energy ODE method.

# get bent Rupinder Matharu

import random
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import copy

import time

#############################################################################
#############################################################################


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


#############################################################################
#############################################################################


# Creates list with start and end coordinates of a unit line


def random_edge_uniform(L):
    # Get coordinates of a random node in the box [0, L] x [0, L]
    initial_x = random.uniform(0, L)
    initial_y = random.uniform(0, L)

    # Obtain a random angle from 0 to pi
    theta = random.uniform(0, math.pi)

    # Obtain end points of the edge
    end_x = initial_x + math.cos(theta)
    end_y = initial_y + math.sin(theta)

    # Return the node positions
    return [[initial_x, initial_y], [end_x, end_y]]


#############################################################################
#############################################################################


def intersection_line_old(input_line1, input_line2):
    try:  # avoids divide by zero in horizontal line case
        m_1 = (input_line1[1][1] - input_line1[1][0]) / (input_line1[0][1] - input_line1[0][0])
        c_1 = input_line1[1][0] - m_1 * input_line1[0][0]
    except ZeroDivisionError:
        m_1 = 0
        c_1 = input_line1[1][0]

    try:  # avoids divide by zero in horizontal line case
        m_2 = (input_line2[1][1] - input_line2[1][0]) / (input_line2[0][1] - input_line2[0][0])
        c_2 = input_line2[1][0] - m_2 * input_line2[0][0]
    except ZeroDivisionError:
        m_2 = 0
        c_2 = input_line2[1][0]
    try:  # returns None if both lines are vertical
        x_intersection = (c_1 - c_2) / (m_2 - m_1)
        y_intersection = m_1 * x_intersection + c_1
        return [x_intersection, y_intersection]
    except ZeroDivisionError:
        return None


# takes in arrays defining line segments and returns itersection points
def intersection_line(Q, P):
    V = Q[0] - P[0]
    R = P[1] - P[0]
    S = Q[1] - Q[0]
    t = np.cross(V, S) / np.cross(R, S)
    return P[0] + t * R


#############################################################################
#############################################################################

# Checks if a line defined by 2 coords intersects a boundary and returns the coordinate
# of intersection


def intersection_boundary(node_1, node_2, L):
    # Random orientations chosen such that the initial y coordinate is always below the end y
    #  coordinate so here we only check if we intersect the top of the boundary for the y coordinate
    if intersect(node_1, node_2, [0, L], [L, L]):
        # skip divide by zero errors
        try:
            m = (node_2[1] - node_1[1]) / (node_2[0] - node_1[0])
            if node_2[0] != L:
                return [node_1[0] + (L - node_1[1]) / m, L]
        except ZeroDivisionError:
            pass
    x_intercept = None
    # intersection with x = 0 boundary
    if intersect(node_1, node_2, [0, 0], [0, L]):
        if node_1[0] != 0:
            x_intercept = 0
    # intersection with x=L boundary
    elif intersect(node_1, node_2, [L, 0], [L, L]):
        if node_1[0] != L:
            x_intercept = L
    # Skip included for when one of the nodes is allready on the boundary
    if x_intercept is None:
        return
    # Calculate y coordinate of intersection
    try:
        intercept_parameter = (x_intercept - node_1[0]) / (node_2[0] - node_1[0])
    except ZeroDivisionError:
        return

    # Return intersection point with vertical wall
    return [x_intercept, node_1[1] + intercept_parameter * (node_2[1] - node_1[1])]


# Takes in two coordinates and returns a list of lists containing line segments


def apply_pbc(node_1, node_2, L):
    boundary_node_1 = intersection_boundary(node_1, node_2, L)

    if boundary_node_1 is None:  # Edge doesnt intersect boundary, exit function
        return [[node_1, node_2]]

    if any(
        [abs((item - L)) <= 1e-15 for item in boundary_node_1]
    ):  # edge intesects top or right boundary
        boundary_node_2 = list(np.mod(boundary_node_1, L))
        index = np.nonzero(boundary_node_2)[0][0] - 1
        node_2[index] = np.mod(node_2[index], L)

    else:  # edge intersects left or right boundary
        index = np.nonzero(boundary_node_1)[0][0] - 1
        boundary_node_2 = copy.deepcopy(boundary_node_1)
        boundary_node_2[index] = L
        node_2[index] = node_2[index] + L

    output = [[node_1, boundary_node_1]]

    output += apply_pbc(boundary_node_2, node_2, L)

    return output


#############################################################################
#############################################################################


def Create_pbc_Network(
    L,
    density,
    seed,
):  # positions_distribution="uniform", orientation_distribrution="uniform"):
    random.seed(seed)

    lines = []
    nodes = []
    edges = []

    line_is_on_boundary = []
    edge_is_on_boundary = []

    N = int(5.637 * density * L**2)

    for i in range(N):  # generate a bunch of line segements under PBC
        line = random_edge_uniform(L)
        line = apply_pbc(line[0], line[1], L)
        if len(line) > 2:
            for j in range(len(line)):
                line_is_on_boundary.append(1)
        else:
            line_is_on_boundary.append(0)
            line_is_on_boundary.append(0)

        lines = lines + line

    lines = [np.array(item) for item in lines]
    nodes = [item for line in lines for item in line]

    intersections = []
    intersections_ordering = []  # order of when lines intersect with eachother

    crosslink_coordinates = []
    num_intersections_per_edge = []
    cumsum_num_intersections_per_edge = []

    for i in range(len(lines)):
        intersections.append([])
        intersections_ordering.append([])
        crosslink_coordinates.append([])

    # This will be a list of all non-zero elements of A_jk
    unsigned_incidence_matrix_list = []

    for (current_line_index, current_line) in enumerate(lines):

        flag_intersection_with_previous_line = 0
        added_nodes = []

        if len(intersections[current_line_index]) != 0:
            flag_intersection_with_previous_line = 1
            added_nodes = [item for item in intersections[current_line_index] if item[0] > item[1]]

        current_node = current_line[0]

        # Run intersection check over other elements of the list ignoring duplicates

        for (line_index, other_line) in enumerate(lines[current_line_index + 1 :]):
            if intersect(current_line[0], current_line[1], other_line[0], other_line[1]):

                # For efficiency we loop over i,j>i, and when line i intersects line j, we note that
                # line j intersects line i in the appropriate place.

                # Find out which lines intersect each other and store pairs of indices

                intersections[current_line_index].append(
                    [current_line_index, current_line_index + line_index + 1]
                )
                intersections[current_line_index + line_index + 1].append(
                    [current_line_index + line_index + 1, current_line_index]
                )

                # Compute coordinates of the crosslink and store it

                crosslink = intersection_line(current_line, other_line)

                crosslink_coordinates[current_line_index].append(crosslink)
                crosslink_coordinates[current_line_index + line_index + 1].append(crosslink)

                # Find distance from start of the line to *this* crosslink and add to a list

                intersections_ordering[current_line_index].append(
                    np.linalg.norm(current_node - crosslink)
                )
                intersections_ordering[current_line_index + line_index + 1].append(
                    np.linalg.norm(current_node - crosslink)
                )

        num_intersections_per_edge.append(len(intersections[current_line_index]))
        cumsum_num_intersections_per_edge.append(sum(num_intersections_per_edge))

        # Use the list of crosslink distances to determine the order of the crosslinks
        # Goes (start of line, crosslink 1, crosslink 2, ......, crosslink N, end of line)

        indices_of_ordered_intersections = np.argsort(intersections_ordering[current_line_index])

        # Order the crosslinks and the intersections such that they follow the above
        # order when read left to right.

        intersections[current_line_index] = [
            intersections[current_line_index][item] for item in indices_of_ordered_intersections
        ]
        crosslink_coordinates[current_line_index] = [
            crosslink_coordinates[current_line_index][item]
            for item in indices_of_ordered_intersections
        ]

        # The index of nodes that do not arise from crosslinks in the list of nodes is given by
        # start_index = 2*current_line_index, and end_index = 2*current_line_index+1

        if len(intersections[current_line_index]) == 0:
            edges.append(current_line)
            unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index])
            unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index + 1])

            # Check if the edge is incident with the boundary, and store that information

            if line_is_on_boundary[current_line_index]:
                edge_is_on_boundary.append([1])
            else:
                edge_is_on_boundary.append([0])

        else:

            # Case where crosslink nodes have not allready been added to list of nodes
            if not flag_intersection_with_previous_line:
                coordinates_list = crosslink_coordinates[current_line_index]
                # The first edge

                edges.append(np.array([current_line[0], coordinates_list[0]]))

                # Check if the edge is incident with the boundary, and store that information

                if line_is_on_boundary[current_line_index] and any(
                    [any(np.mod(item, L) == 0) for item in edges[-1]]
                ):
                    edge_is_on_boundary.append(1)
                else:
                    edge_is_on_boundary.append(0)

                unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index])
                unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes)])
                nodes.append(coordinates_list[0])

                # Loop over all the edges made of incident crosslinks
                for i in range(len(coordinates_list) - 1):

                    edges.append(np.array([coordinates_list[i], coordinates_list[i + 1]]))
                    edge_is_on_boundary.append(0)

                    unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes)])
                    unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes) + 1])

                    nodes.append(coordinates_list[i + 1])

                # The final edge

                edges.append(np.array([coordinates_list[-1], current_line[1]]))

                # Check if the edge is incident with the boundary, and store that information

                if line_is_on_boundary[current_line_index] and any(
                    [any(np.mod(item, L) == 0) for item in edges[-1]]
                ):
                    edge_is_on_boundary.append(1)
                else:
                    edge_is_on_boundary.append(0)

                unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index])
                unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes)])

            else:  # Case where crosslink nodes have allready been added to list of nodes
                coordinates_list = crosslink_coordinates[current_line_index]

                added_nodes_indices = []

                for item in added_nodes:

                    item_rev = copy.copy(item)
                    item_rev.reverse()

                    index = cumsum_num_intersections_per_edge[item[1] - 1] + intersections[
                        item[1]
                    ].index(item_rev)

                    added_nodes_indices.append(index + 2 * len(lines))

                # The first edge

                edges.append(np.array([current_line[0], coordinates_list[0]]))

                # Check if the edge is incident with the boundary, and store that information

                if line_is_on_boundary[current_line_index] and any(
                    [any(np.mod(item, L) == 0) for item in edges[-1]]
                ):
                    edge_is_on_boundary.append(1)
                else:
                    edge_is_on_boundary.append(0)

                if added_nodes[0] == intersections[current_line_index][0]:
                    unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index])
                    unsigned_incidence_matrix_list.append([len(edges) - 1, added_nodes_indices[0]])

                else:
                    unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index])
                    unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes)])

                    nodes.append(coordinates_list[0])

                # Loop over all the edges made of incident crosslinks
                for i in range(len(coordinates_list) - 1):
                    edges.append(np.array([coordinates_list[i], coordinates_list[i + 1]]))

                    # Check if the edge is incident with the boundary, and store that information

                    if line_is_on_boundary[current_line_index] and any(
                        [any(np.mod(item, L) == 0) for item in edges[-1]]
                    ):
                        edge_is_on_boundary.append(1)
                    else:
                        edge_is_on_boundary.append(0)

                    index_node_1 = None
                    index_node_2 = None

                    # Check if the two nodes have allready been added to the list.

                    if intersections[current_line_index][i] in added_nodes:
                        index_node_1 = added_nodes_indices[
                            added_nodes.index(intersections[current_line_index][i])
                        ]

                    if intersections[current_line_index][i + 1] in added_nodes:
                        index_node_2 = added_nodes_indices[
                            added_nodes.index(intersections[current_line_index][i + 1])
                        ]

                    # Check the 4 cases, both nodes are added, one is added, neither have been added

                    if index_node_1 is not None and index_node_2 is not None:
                        unsigned_incidence_matrix_list.append([len(edges) - 1, index_node_1])
                        unsigned_incidence_matrix_list.append([len(edges) - 1, index_node_2])

                    elif index_node_1 is not None:
                        unsigned_incidence_matrix_list.append([len(edges) - 1, index_node_1])
                        unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes)])
                        nodes.append(coordinates_list[i + 1])

                    elif index_node_2 is not None:
                        unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes)])
                        unsigned_incidence_matrix_list.append([len(edges) - 1, index_node_2])

                    else:
                        unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes)])
                        unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes) + 1])
                        nodes.append(coordinates_list[i + 1])

                # The final edge

                edges.append(np.array([coordinates_list[-1], current_line[1]]))

                # Check if the edge is incident with the boundary, and store that information

                if line_is_on_boundary[current_line_index] and any(
                    [any(np.mod(item, L) == 0) for item in edges[-1]]
                ):
                    edge_is_on_boundary.append(1)
                else:
                    edge_is_on_boundary.append(0)

                if added_nodes[-1] == intersections[current_line_index][-1]:
                    unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index])
                    unsigned_incidence_matrix_list.append([len(edges) - 1, added_nodes_indices[-1]])

                else:
                    unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index])
                    unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes)])

    # for (line_index, line) in lines:
    #     if len(intersections[line_index]) == 0:
    #         edges.append(line)
    #     else:
    #         coordinates_list = (
    #             [line[0]] + [item for item in crosslink_coordinates[line_index]] + line[1]
    #         )
    #         for i in range(len(coordinates_list) - 1):
    #             edges.append(np.array([coordinates_list[i], coordinates_list[i + 1]]))
    #         for item in coordinates_list:
    #             nodes.append(item)

    # boundary_nodes = []
    # boundary_indices = []
    # for (i, entry) in enumerate(lines):
    #     for (j, element) in enumerate(entry):
    #         if any(np.mod(element, L) <= 1e-15):
    #             boundary_nodes.append(entry)
    #             boundary_indices.append([i, j])

    # nodes_list = []  # We now create a list of all the nodes
    # for item in lines:
    #     nodes_list.append(item[0])
    #     nodes_list.append(item[1])

    # for item in crosslink_coordinates:
    #     nodes_list.append(item)

    # intersection_sets = []
    # for (index, item) in enumerate(intersections):
    #     for element in item:
    #         intersection_sets.append(set([2 * element, index + 2 * len(lines)]))
    #         intersection_sets.append(set([2 * element + 1, index + 2 * len(lines)]))

    # # Here is where we then figure out all the new lines
    # edges = []
    # num_intersections = 1
    # for index in range(len(intersections)):
    #     if num_intersections > 1:
    #         num_intersections -= 1
    #         continue
    #     item = intersections[index]
    #     try:
    #         while item[0] == intersections[index + num_intersections][0]:
    #             num_intersections += 1
    #     except IndexError:
    #         pass
    #     print(index, item[0], num_intersections)
    #     distances = [
    #         np.linalg.norm(lines[item[0]][0] - crosslink_coordinates[intersections[index + i][1]])
    #         for i in range(num_intersections)
    #     ]
    #     coord_store = [
    #         crosslink_coordinates[intersections[index + i][1]] for i in range(num_intersections)
    #     ]
    #     for item in coord_store:

    #         edges.append(
    #             np.array[
    #                 lines[item[0]][0],
    #             ]
    #         )

    return (
        lines,
        intersections,
        intersections_ordering,
        crosslink_coordinates,
        nodes,
        unsigned_incidence_matrix_list,
        edges,
        line_is_on_boundary,
        edge_is_on_boundary,
    )


def PlotNetwork(lines, L):
    fig_network = plt.figure()

    for segment in lines:
        plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color="b")

    # Plot the box
    plt.plot([0, L, L, 0, 0], [0, 0, L, L, 0])

    plt.xlim(0 - 0.1 * L, 1.1 * L)
    plt.ylim(0 - 0.1 * L, 1.1 * L)

    plt.gca().set_aspect("equal")
    plt.show
    return
