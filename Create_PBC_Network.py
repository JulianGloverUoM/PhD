# -*- coding: utf-8 -*-

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
import scipy as sp
import scipy.stats as stats
from scipy.sparse import lil_matrix
from datetime import date
import copy
import time
import os
import sys
import pickle

file_path = os.path.realpath(__file__)
sys.path.append(file_path)


#############################################################################
#############################################################################


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def slope(A, B):
    return (A[1] - B[1]) / (A[0] - B[0])


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


def trim_rows(matrix, min_nonzeros):
    non_empty_rows = []
    removed_indices = []

    for i in range(matrix.shape[0]):
        if len(matrix.rows[i]) >= min_nonzeros:
            non_empty_rows.append(i)
        else:
            removed_indices.append(i)

    new_matrix = lil_matrix((len(non_empty_rows), matrix.shape[1]))
    for new_index, old_index in enumerate(non_empty_rows):
        new_matrix.rows[new_index] = matrix.rows[old_index]
        new_matrix.data[new_index] = matrix.data[old_index]
    return new_matrix, removed_indices


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
        original_line = copy.deepcopy(line)

        line = apply_pbc(line[0], line[1], L)
        if line != original_line:
            for j in range(len(line)):
                line_is_on_boundary.append(1)
        else:
            line_is_on_boundary.append(0)

        lines = lines + line

    lines = [np.array(item) for item in lines]
    nodes = [item for line in lines for item in line]

    intersections = []
    intersections_ordering = []  # order of when lines intersect with eachother

    crosslink_coordinates = []
    num_intersections_per_line = []
    cumsum_num_intersections_per_line = [0]

    for i in range(len(lines)):
        intersections.append([])
        intersections_ordering.append([])
        crosslink_coordinates.append([])

    # This will be a list of all non-zero elements of A_jk
    unsigned_incidence_matrix_list = []

    for (current_line_index, current_line) in enumerate(lines):

        current_node = current_line[0]

        flag_intersection_with_previous_line = 0
        added_nodes = []

        if len(intersections[current_line_index]) != 0:
            flag_intersection_with_previous_line = 1
            added_nodes = [item for item in intersections[current_line_index] if item[0] > item[1]]
            for item in crosslink_coordinates[current_line_index]:
                intersections_ordering[current_line_index].append(
                    np.linalg.norm(current_node - item)
                )

        # Run intersection check over other elements of the list ignoring duplicates

        for (line_index, other_line) in enumerate(lines[current_line_index + 1 :]):
            if intersect(current_line[0], current_line[1], other_line[0], other_line[1]):

                # For efficiency we loop over i,j>i, and when line i intersects line j, we record
                # that line j intersects line i in the appropriate place.

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

        num_intersections_per_line.append(
            len([item for item in intersections[current_line_index] if item[0] < item[1]])
        )
        cumsum_num_intersections_per_line.append(sum(num_intersections_per_line))

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
            if line_is_on_boundary[current_line_index]:
                edges.append(current_line)
                unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index])
                unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index + 1])
                edge_is_on_boundary.append(1)
        else:

            # Case where crosslink nodes have not allready been added to list of nodes
            if not flag_intersection_with_previous_line:
                coordinates_list = crosslink_coordinates[current_line_index]
                # The first edge

                # Check if the edge is incident with the boundary, and store that information

                edges.append(np.array([current_line[0], coordinates_list[0]]))
                unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index])
                unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes)])

                if line_is_on_boundary[current_line_index] and any(
                    [any(np.mod(item, L) == 0) for item in [current_line[0], coordinates_list[0]]]
                ):
                    edge_is_on_boundary.append(1)
                else:
                    edge_is_on_boundary.append(0)

                nodes.append(coordinates_list[0])

                # Loop over all the edges made of incident crosslinks
                for i in range(len(coordinates_list) - 1):

                    edges.append(np.array([coordinates_list[i], coordinates_list[i + 1]]))
                    edge_is_on_boundary.append(0)

                    unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes) - 1])
                    unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes)])

                    nodes.append(coordinates_list[i + 1])

                # The final edge

                edges.append(np.array([coordinates_list[-1], current_line[1]]))
                unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index + 1])
                unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes) - 1])
                # Check if the edge is incident with the boundary, and store that information

                if line_is_on_boundary[current_line_index] and any(
                    [any(np.mod(item, L) == 0) for item in [coordinates_list[-1], current_line[1]]]
                ):

                    edge_is_on_boundary.append(1)
                else:
                    edge_is_on_boundary.append(0)

            else:  # Case where crosslink nodes have allready been added to list of nodes
                coordinates_list = crosslink_coordinates[current_line_index]

                added_nodes_indices = []

                for item in added_nodes:

                    item_rev = copy.copy(item)
                    item_rev.reverse()

                    item_index = intersections[item[1]].index(item_rev)

                    index = cumsum_num_intersections_per_line[item[1]] + item_index

                    index_discount_for_added_nodes = len(
                        [elem for elem in intersections[item[1]][:item_index] if elem[0] > elem[1]]
                    )

                    added_nodes_indices.append(
                        index + 2 * len(lines) - index_discount_for_added_nodes
                    )

                # The first edge

                edges.append(np.array([current_line[0], coordinates_list[0]]))

                if intersections[current_line_index][0] in added_nodes:
                    unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index])
                    unsigned_incidence_matrix_list.append(
                        [
                            len(edges) - 1,
                            added_nodes_indices[
                                added_nodes.index(intersections[current_line_index][0])
                            ],
                        ]
                    )

                else:
                    unsigned_incidence_matrix_list.append([len(edges) - 1, 2 * current_line_index])
                    unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes)])

                    nodes.append(coordinates_list[0])

                # Check if the edge is incident with the boundary, and store that information

                if line_is_on_boundary[current_line_index] and any(
                    [any(np.mod(item, L) == 0) for item in [current_line[0], coordinates_list[0]]]
                ):

                    edge_is_on_boundary.append(1)
                else:
                    edge_is_on_boundary.append(0)

                # Loop over all the edges made of incident crosslinks
                for i in range(len(coordinates_list) - 1):
                    edges.append(np.array([coordinates_list[i], coordinates_list[i + 1]]))
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
                        unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes) - 1])
                        unsigned_incidence_matrix_list.append([len(edges) - 1, index_node_2])

                    else:
                        unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes) - 1])
                        unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes)])
                        nodes.append(coordinates_list[i + 1])

                # The final edge

                edges.append(np.array([coordinates_list[-1], current_line[1]]))

                if intersections[current_line_index][-1] in added_nodes:
                    unsigned_incidence_matrix_list.append(
                        [len(edges) - 1, 2 * current_line_index + 1]
                    )
                    unsigned_incidence_matrix_list.append(
                        [
                            len(edges) - 1,
                            added_nodes_indices[
                                added_nodes.index(intersections[current_line_index][-1])
                            ],
                        ]
                    )

                else:
                    unsigned_incidence_matrix_list.append(
                        [len(edges) - 1, 2 * current_line_index + 1]
                    )
                    unsigned_incidence_matrix_list.append([len(edges) - 1, len(nodes) - 1])

                # Check if the edge is incident with the boundary, and store that information

                if line_is_on_boundary[current_line_index] and any(
                    [any(np.mod(item, L) == 0) for item in [coordinates_list[-1], current_line[1]]]
                ):

                    edge_is_on_boundary.append(1)
                else:
                    edge_is_on_boundary.append(0)

    # Here we create the initial incidence matrix and set it up, although it contains boundary nodes.
    incidence_matrix = lil_matrix((len(edges), len(nodes)))
    for i in range(len(edges)):
        index_1 = unsigned_incidence_matrix_list[2 * i]
        index_2 = unsigned_incidence_matrix_list[2 * i + 1]
        incidence_matrix[index_1[0], index_1[1]] = 1
        incidence_matrix[index_2[0], index_2[1]] = -1

    # Now we loop over the nodes, determine which ones are on the boundary and find their associated
    # edges and connections. We do this to then be able to alter the incidence matrix such that it
    # is the matrix of the periodic structure, and identify boundary nodes that much be removed from
    # the list.

    boundary_index_list = []
    for i, node in enumerate(nodes):
        if any([abs(item - 0) <= 1e-15 for item in node]) or any(
            [abs(item - L) <= 1e-15 for item in node]
        ):
            edge_index = incidence_matrix.T.rows[i][0]
            boundary_index_list.append([i, edge_index])

    # We convert the boundary index list into a np.array for easier use in the following code.
    boundary_index_list = np.array(boundary_index_list)

    # Now we generate the edge corections. We need this to be able to calculate correct edge vectors
    # and edge lengths in a concise manner. The formula is l_j = A_jk.dot(r_k)+edge_corrections_j.
    edge_corrections = np.zeros((np.shape(incidence_matrix)[0], 2))

    # Some boundary edges cross through one or two boundaries. Due to how the boundary edge list is
    # structured, all the segements consiting an edge are next to each other in the list, so if an
    # edge is made of 3 segments, we can simple add 3 to the boundary edge count, or two otherwise.
    # If we instead looped over edges and assigned the value of our loop variable, bugs can occur.

    boundary_edge_count = 0
    while 2 * boundary_edge_count < len(boundary_index_list):
        # This tests to see if the edge is made of 3 segments. The first check is there to make sure
        # that if the last edge is made of 2 segments, an index error is not raised due to trying to
        # access an element of the boundary_index_list that doesnt exist.
        # Because the python "and" function is lazy, that means if the last boundary edge consists
        # of only 2 segments, the first if statement fails, so the check after the "and" never gets
        # called, and no index error is raised.
        if (2 * boundary_edge_count + 2) < len(boundary_index_list) and (
            boundary_index_list[2 * boundary_edge_count + 1][1]
            == boundary_index_list[2 * boundary_edge_count + 2][1]
        ):
            # We assign the edges for readability.
            edge_1, edge_2, edge_3 = (
                boundary_index_list[2 * boundary_edge_count][1],
                boundary_index_list[2 * boundary_edge_count + 1][1],
                boundary_index_list[2 * boundary_edge_count + 3][1],
            )
            # We determine the nodes for the new edge.
            for item in incidence_matrix.rows[edge_1]:
                if item != boundary_index_list[2 * boundary_edge_count][1]:
                    node_1 = item
            for item in incidence_matrix.rows[edge_3]:
                if item != boundary_index_list[2 * boundary_edge_count + 3][1]:
                    node_2 = item

            # We force the incidence matrix to have no entries where there used to be boundary
            # segments, deleting the boundary edges. To avoid issues with changing indices, we do
            # not delete the edges yet, these empty edges will be removed during a later step.
            incidence_matrix.rows[edge_2] = []
            incidence_matrix.data[edge_2] = []
            incidence_matrix.rows[edge_3] = []
            incidence_matrix.data[edge_3] = []
            # We choose the first edge index to become the new edge index of the new boundary edge.
            incidence_matrix.rows[edge_1] = [node_1, node_2]
            incidence_matrix.data[edge_1] = [1, -1]
            # This check makes sure that the edge corrections is right, there are 4 possibilities,
            # for all the combinations of boundaries that the edge can cross, Left|Right and Top|Bot.
            try:
                if slope(nodes[node_1], nodes[node_2]) > 0:
                    if np.linalg.norm(incidence_matrix[edge_1, :].dot(nodes) + L) <= 1:
                        edge_corrections[edge_1] = L
                    else:
                        edge_corrections[edge_1] = -L
                elif np.linalg.norm(incidence_matrix[edge_1, :].dot(nodes) + [L, -L]) <= 1:
                    edge_corrections[edge_1] = [L, -L]
                else:
                    edge_corrections[edge_1] = [-L, L]
            except:
                print(nodes[node_1], nodes[node_2])
            boundary_edge_count += 2

        else:
            # If a boundary edge consists of two segments, we run essentially the exact same code
            # as in the previous if statement, but with just two edges.
            # We assign the edges for readability.
            edge_1, edge_2 = (
                boundary_index_list[2 * boundary_edge_count][1],
                boundary_index_list[2 * boundary_edge_count + 1][1],
            )
            # We determine the nodes for the new edge.
            for item in incidence_matrix.rows[edge_1]:
                if item != boundary_index_list[2 * boundary_edge_count][1]:
                    node_1 = item
            for item in incidence_matrix.rows[edge_2]:
                if item != boundary_index_list[2 * boundary_edge_count + 1][1]:
                    node_2 = item
            # We force the incidence matrix to have no entries where there used to be boundary
            # segments, deleting the boundary edges. To avoid issues with changing indices, we do
            # not delete the edges yet, these empty edges will be removed during a later step.
            incidence_matrix.rows[edge_1] = []
            incidence_matrix.data[edge_1] = []
            incidence_matrix.rows[edge_2] = []
            incidence_matrix.data[edge_2] = []
            # We choose the first edge index to become the new edge index of the new boundary edge.
            incidence_matrix.rows[edge_1] = [node_1, node_2]
            incidence_matrix.data[edge_1] = [1, -1]
            # As there is only a single boundary that is crossed, we can determine exactly which one
            # it is easily, and so can avoid the more complicated check of the 4 possibilities.
            crossed_boundary_index = (
                np.where(
                    nodes[boundary_index_list[2 * boundary_edge_count][0]]
                    - nodes[boundary_index_list[2 * boundary_edge_count + 1][0]]
                    == 0
                )[0][0]
                - 1
            )
            edge_corrections[edge_1][crossed_boundary_index] = L
            # We assume we add L to the node (in the right coordinate), and if that results in a
            # edge that is too long, that must mean we need to subtract it instead.
            if (
                not np.linalg.norm(
                    incidence_matrix[edge_1, :].dot(nodes) + edge_corrections[edge_1]
                )
                <= 1
            ):
                edge_corrections[edge_1] = -1 * edge_corrections[edge_1]

            boundary_edge_count += 1

    # This is an initial trimming step, we seperate it from the loop as we know what nodes to delete

    boundary_index_list_reverse = list(copy.deepcopy(boundary_index_list))

    boundary_index_list_reverse.reverse()

    deleted_nodes = []
    for item in boundary_index_list_reverse:
        del nodes[item[0]]
        deleted_nodes.append(item[0])

    deleted_nodes = set(deleted_nodes)

    num_edges = len(incidence_matrix.rows)
    num_nodes = len(nodes)
    new_matrix = lil_matrix((num_nodes, num_edges))

    incidence_matrix_transpose = copy.deepcopy(incidence_matrix.T)

    for i in range(num_nodes):
        if i not in deleted_nodes:
            for j in range(len(incidence_matrix_transpose.data[i])):
                new_matrix[
                    i, incidence_matrix_transpose.rows[i][j]
                ] = incidence_matrix_transpose.data[i][j]

    incidence_matrix = copy.deepcopy(new_matrix.T)
    # In Trimming we remove any empty edges or dangling nodes we have not already identified. This
    # could be done in a single pass of the incidence matrix, but as deleting an item may lead to a
    # new item requiring deletion, its easier to just pass over the list multiple times.
    # A technically more efficient code may be to identify and check if a edges deletion should
    # result in new edges and nodes being deleted, but doing so would require some form of recursion,
    # and the code would be more prone to bugs and harder to maintain.
    # Instead we use a while loop, run over the incidence matrix, delete edges and nodes as they
    # arise, and then loop over the incidence matrix repeated until nothing is deleted, then halts.
    trimming = True
    while trimming:
        # Save initial shape
        initial_shape = incidence_matrix.shape

        # Step 1: Trim dangle or empty edges
        incidence_matrix, removed_edges = trim_rows(incidence_matrix, min_nonzeros=2)
        for edge_index in reversed(removed_edges):
            edge_corrections = np.delete(edge_corrections, edge_index, axis=0)
        # Step 2: Transpose
        incidence_matrix = incidence_matrix.T

        # Step 3: Trim rows (columns of original matrix) with less than 1 non-zero entry
        incidence_matrix, removed_nodes = trim_rows(incidence_matrix, min_nonzeros=1)

        for node_index in reversed(removed_nodes):
            del nodes[node_index]

        # Step 4: Transpose back
        incidence_matrix = incidence_matrix.T

        # Check if dimensions have changed
        if incidence_matrix.shape == initial_shape:
            incidence_matrix_csr = incidence_matrix.tocsr()
            trimming = False

    return (
        nodes,
        edge_corrections,
        incidence_matrix_csr,
    )


# doesnt work


def ColormapPlot_PBC_dilation(
    nodes, incidence_matrix, L, lambda_1, lambda_2, initial_lengths, edge_corrections
):
    strains = (
        vector_of_magnitudes(incidence_matrix.dot(nodes) + edge_corrections) - 0.9 * initial_lengths
    ) / initial_lengths

    cm1 = mcol.LinearSegmentedColormap.from_list("bpr", ["b", "r"])
    cnorm = mcol.Normalize(vmin=min(strains), vmax=max(strains))
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])
    fig = plt.figure()
    plt.title(str(r"$\lambda_1,\lambda_2$ = {},{}".format(lambda_1, lambda_2)))
    plt.gca().set_aspect("equal")

    plotting_edges = []
    for row_index, row in enumerate(incidence_matrix):
        row_correction = edge_corrections[row_index]
        if any(row_correction):
            node_1_index, node_2_index = np.nonzero(row)[0]
            if row[node_1_index] != 1:
                node_1_index, node_2_index = node_2_index, node_1_index
            node_1 = nodes[node_1_index] + row_correction
            node_2 = nodes[node_2_index]
            if apply_pbc(node_1, node_2, L)[0] == [
                node_1,
                node_2,
            ]:
                node_1 = nodes[node_1_index]
                node_2 = nodes[node_2_index] - row_correction
            plotting_edges.append(apply_pbc(node_1, node_2, L))
        else:
            node_1 = list(nodes[np.nonzero(row)[0][0]])
            node_2 = list(nodes[np.nonzero(row)[0][1]])
            plotting_edges.append([node_1, node_2])

    for i, edge in enumerate(plotting_edges):
        if type(edge[0][0]) == np.float64:
            plt.plot(
                [edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color=cpick.to_rgba(strains[i])
            )
        else:
            for segment in edge:
                plt.plot(
                    [segment[0][0], segment[1][0]],
                    [segment[0][1], segment[1][1]],
                    color=cpick.to_rgba(strains[i]),
                )

    plt.plot(
        [0, lambda_1 * L, lambda_1 * L, 0, 0],
        [0, 0, lambda_2 * L, lambda_2 * L, 0],
    )
    ax = plt.gca()

    # plt.xlim(0 - 0.1 * L, 1.1 * lambda_1 * L)
    # plt.ylim(0 - 0.1 * L, 1.1 * lambda_2 * L)

    plt.colorbar(
        cpick,
        cax=fig.add_axes([0.85, 0.25, 0.05, 0.5]),
        boundaries=np.arange(min(strains), max(strains), (max(strains) - min(strains)) / 100),
    )
    # plt.savefig("prestress_network_deformed_equilibrium_example.pdf")

    return


#############################################################################
