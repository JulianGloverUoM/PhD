#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:08:06 2025

@author: v17847jg
"""

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
    # coordinate so here we only check if we intersect the top of the boundary for the y coordinate
    if intersect(node_1, node_2, [0, L], [L, L]):
        # skip divide by zero errors
        try:
            m = (node_2[1] - node_1[1]) / (node_2[0] - node_1[0])
            if node_2[0] != L:
                return [node_1[0] + (L - node_1[1]) / m, L]
        except ZeroDivisionError:
            pass

    # In generation we do not need to check for intersection with the bottom boundary, however, to
    # plot the networks we will need to check for intersections with the bottom boundary and so
    # include the below check.

    if node_1[1] != 0 and intersect(node_1, node_2, [0, 0], [L, 0]):
        # skip divide by zero errors
        try:
            m = (node_2[1] - node_1[1]) / (node_2[0] - node_1[0])
            if node_2[0] != 0:
                return [node_1[0] - (node_1[1] / m), 0]
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


def trim_nodes(matrix, min_nonzeros, nodes, L):
    non_empty_rows = []
    removed_indices = []

    for i in range(matrix.shape[0]):
        if len(matrix.rows[i]) == 0:
            removed_indices.append(i)
        elif (
            any([abs(item - 0) <= 1e-15 for item in nodes[i]])
            or any([abs(item - L) <= 1e-15 for item in nodes[i]])
        ) or len(matrix.rows[i]) >= min_nonzeros:
            non_empty_rows.append(i)
        else:
            removed_indices.append(i)

    new_matrix = lil_matrix((len(non_empty_rows), matrix.shape[1]))
    for new_index, old_index in enumerate(non_empty_rows):
        new_matrix.rows[new_index] = matrix.rows[old_index]
        new_matrix.data[new_index] = matrix.data[old_index]
    return new_matrix, removed_indices


def permute_sparse_matrix(M, new_row_order=None, new_col_order=None):
    """
    Reorders the rows and/or columns in a scipy sparse matrix
        using the specified array(s) of indexes
        e.g., [1,0,2,3,...] would swap the first and second row/col.
    """
    if new_row_order is None and new_col_order is None:
        return M

    new_M = M
    if new_row_order is not None:
        I = sp.sparse.eye(M.shape[0]).tocoo()
        I.row = I.row[new_row_order]
        new_M = I.dot(new_M)
    if new_col_order is not None:
        I = sp.sparse.eye(M.shape[1]).tocoo()
        I.col = I.col[new_col_order]
        new_M = new_M.dot(I)
    return new_M


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
        # print(initial_shape)

        # Step 1: Trim dangle or empty edges
        incidence_matrix, removed_edges = trim_rows(incidence_matrix, min_nonzeros=2)
        # print(removed_edges)

        # Step 2: Transpose
        incidence_matrix = incidence_matrix.T

        # Step 3: Trim rows (columns of original matrix) with less than 2 non-zero entry
        incidence_matrix, removed_nodes = trim_nodes(incidence_matrix, 2, nodes, L)
        # print(removed_nodes)

        for node_index in reversed(removed_nodes):
            del nodes[node_index]

        # Step 4: Transpose back
        incidence_matrix = incidence_matrix.T
        # print(incidence_matrix.shape)
        # Check if dimensions have changed
        if incidence_matrix.shape == initial_shape:
            incidence_matrix_csr = incidence_matrix.tocsr()
            trimming = False

    nodes_copy = copy.deepcopy(nodes)
    count_of_swapped_nodes = 1
    num_nodes = len(nodes)
    new_col_order = [i for i in range(num_nodes)]
    for i, node in enumerate(nodes_copy):
        if any([abs(item - 0) <= 1e-15 for item in node]) or any(
            [abs(item - L) <= 1e-15 for item in node]
        ):
            swapped_index = num_nodes - count_of_swapped_nodes
            nodes[swapped_index] = node
            nodes[i] = nodes_copy[swapped_index]

            new_col_order[i] = swapped_index
            new_col_order[swapped_index] = i

            count_of_swapped_nodes += 1

    incidence_matrix = permute_sparse_matrix(incidence_matrix, None, new_col_order)

    incidence_matrix_csr = incidence_matrix.tocsr()

    boundary_nodes = num_nodes - count_of_swapped_nodes + 1

    return (
        nodes,
        boundary_nodes,
        incidence_matrix_csr,
    )


#############################################################################


def ColormapPlot_dilation(
    nodes,
    incidence_matrix,
    L,
    lambda_1,
    lambda_2,
    plotted_quantity,
    plotted_quantity_name="Plotted Quantity",
):

    cm1 = mcol.LinearSegmentedColormap.from_list("bpr", ["b", "r"])
    cnorm = mcol.Normalize(vmin=min(plotted_quantity), vmax=max(plotted_quantity))
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])
    fig = plt.figure()
    plt.title(str(r"$\lambda_1,\lambda_2$ = {},{}".format(lambda_1, lambda_2)))
    plt.gca().set_aspect("equal")

    new_edges = []
    for row in range(incidence_matrix.shape[0]):
        index_1, index_2 = incidence_matrix.getrow(row).indices
        node_1 = nodes[index_1]
        node_2 = nodes[index_2]
        new_edges.append([node_1, node_2])

    for i in range(len(new_edges)):
        edge = new_edges[i]

        plt.plot(
            [edge[0][0], edge[1][0]],
            [edge[0][1], edge[1][1]],
            color=cpick.to_rgba(plotted_quantity[i]),
            linewidth=0.3,
        )

    plt.plot(
        [0, lambda_1 * L, lambda_1 * L, 0, 0],
        [0, 0, lambda_2 * L, lambda_2 * L, 0],
    )
    ax = plt.gca()

    # plt.xlim(0 - 0.1 * L, 1.1 * lambda_1 * L)
    # plt.ylim(0 - 0.1 * L, 1.1 * lambda_2 * L)

    cax = fig.add_axes([0.85, 0.25, 0.05, 0.5])
    cbar = plt.colorbar(
        cpick,
        cax=cax,
        boundaries=np.arange(
            min(plotted_quantity),
            max(plotted_quantity),
            min((max(plotted_quantity) - min(plotted_quantity)) / 100, 100),
        ),
    )

    # Set title above colorbar
    cax.set_title(plotted_quantity_name)

    # plt.savefig("Informal_applied_talk_example_7_4.pdf")

    plt.show()

    return
