#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script to generate a PBC network with data structure in compliance with requirements for solving
# for equilibrium positions using dispersive energy ODE method.

# get bent Rupinder Matharu <3

from dataclasses import dataclass
import random
import math
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import copy
import os
import sys
import pickle

file_path = os.path.realpath(__file__)
sys.path.append(file_path)

#############################################################################
#############################################################################


@dataclass(frozen=True)
class Network_Law:
    name: str
    propose_edge: callable


def fibre_angle(edge):
    dx = edge[1][0] - edge[0][0]
    dy = edge[1][1] - edge[0][1]
    return np.mod(np.arctan2(dy, dx), np.pi)


def point_segment_distance(point, segment_start, segment_end):
    point = np.asarray(point, dtype=float)
    segment_start = np.asarray(segment_start, dtype=float)
    segment_end = np.asarray(segment_end, dtype=float)

    segment = segment_end - segment_start
    seg_len_sq = np.dot(segment, segment)

    if seg_len_sq == 0:
        return np.linalg.norm(point - segment_start)

    t = np.clip(
        np.dot(point - segment_start, segment) / seg_len_sq,
        0.0,
        1.0,
    )

    closest = segment_start + t * segment

    return np.linalg.norm(point - closest)


def initialise_segment_grid(L):
    # The deposited fibres have length 1, so we use a grid with unit square cells.
    if int(L) != L:
        raise ValueError("The segment grid assumes that L is an integer.")

    L = int(L)

    return [[[] for j in range(L)] for i in range(L)]


def grid_cell(point, L):
    L = int(L)
    point = np.mod(point, L)

    return int(point[0]) % L, int(point[1]) % L


def segment_grid_cells(segment, L):
    # Returns all grid cells touched by the bounding box of a segment.
    # This over-counts slightly, but keeps the code simple and safe.
    L = int(L)

    segment_start = np.asarray(segment[0], dtype=float)
    segment_end = np.asarray(segment[1], dtype=float)

    x_min = min(segment_start[0], segment_end[0])
    x_max = max(segment_start[0], segment_end[0])
    y_min = min(segment_start[1], segment_end[1])
    y_max = max(segment_start[1], segment_end[1])

    i_min = max(0, int(np.floor(x_min)))
    i_max = min(L - 1, int(np.floor(x_max - 1e-15)))
    j_min = max(0, int(np.floor(y_min)))
    j_max = min(L - 1, int(np.floor(y_max - 1e-15)))

    cells = []

    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            cells.append((i, j))

    # Add endpoint cells as well, which helps with segments lying exactly on a boundary.
    cells.append(grid_cell(segment_start, L))
    cells.append(grid_cell(segment_end, L))

    return list(set(cells))


def add_segment_to_grid(segment_grid, segment, L):
    for i, j in segment_grid_cells(segment, L):
        segment_grid[i][j].append(segment)

    return


def nearby_segments(seed, segment_grid, L, search_radius):
    # Returns possible nearby segments, using periodic indexing of the unit-cell grid.
    # The final distance check is still done exactly by point_segment_distance_periodic.
    L = int(L)
    centre_i, centre_j = grid_cell(seed, L)
    cell_range = max(1, int(np.ceil(search_radius)))

    output = []
    used_segments = set()

    for di in range(-cell_range, cell_range + 1):
        for dj in range(-cell_range, cell_range + 1):
            i = (centre_i + di) % L
            j = (centre_j + dj) % L

            for segment in segment_grid[i][j]:
                segment_id = id(segment)

                if segment_id not in used_segments:
                    output.append(segment)
                    used_segments.add(segment_id)

    return output


def nearby_segments_with_shifts(seed, segment_grid, L, search_radius):
    # Returns possible nearby segments, together with the periodic shift needed to compare
    # them with the candidate seed.
    L = int(L)

    centre_i, centre_j = grid_cell(seed, L)
    cell_range = max(1, int(np.ceil(search_radius)))

    output = []
    used_segments = set()

    for di in range(-cell_range, cell_range + 1):
        for dj in range(-cell_range, cell_range + 1):
            raw_i = centre_i + di
            raw_j = centre_j + dj

            i = raw_i % L
            j = raw_j % L

            # If the neighbouring cell has wrapped around the periodic boundary,
            # shift the segment back into the local image of the seed.
            shift = np.array(
                [
                    (raw_i - i) * L,
                    (raw_j - j) * L,
                ],
                dtype=float,
            )

            for segment in segment_grid[i][j]:
                segment_id = id(segment)

                if segment_id not in used_segments:
                    output.append((segment, shift))
                    used_segments.add(segment_id)

    return output


def add_line_index_to_grid(segment_grid, line_index, segment, L):
    for i, j in segment_grid_cells(segment, L):
        segment_grid[i][j].append(line_index)

    return


def nearby_line_indices(segment, segment_grid, L):
    # Returns indices of lines whose bounding-box cells overlap with the current segment.
    # If two segments intersect, their bounding boxes overlap, so checking shared cells is enough.
    L = int(L)

    output = []
    used_indices = set()

    for i, j in segment_grid_cells(segment, L):
        for line_index in segment_grid[i][j]:
            if line_index not in used_indices:
                output.append(line_index)
                used_indices.add(line_index)

    output.sort()

    return output


def make_network_law(key, matern_radius=0.1, max_attempts=10000):
    if key == "Uniform":
        return Network_Law(
            name="Uniform",
            propose_edge=lambda L, accepted_segments, rng: random_edge_uniform(L, rng),
        )

    elif key == "Matern":
        if matern_radius <= 0:
            raise ValueError("matern_radius must be positive.")

        def propose_edge(L, accepted_segments, rng):
            for _ in range(max_attempts):
                candidate = random_edge_uniform(L, rng)
                seed = candidate[0]

                candidate_segments = nearby_segments_with_shifts(
                    seed,
                    accepted_segments,
                    L,
                    matern_radius,
                )

                if len(candidate_segments) == 0:
                    return candidate

                min_distance = np.inf

                for segment, shift in candidate_segments:
                    distance = point_segment_distance(
                        seed,
                        np.asarray(segment[0]) + shift,
                        np.asarray(segment[1]) + shift,
                    )

                    if distance < min_distance:
                        min_distance = distance

                if min_distance >= matern_radius:
                    return candidate

                rejection_probability = 1.0 - min_distance / matern_radius

                if rng.random() >= rejection_probability:
                    return candidate

            raise RuntimeError(
                "Could not place a fibre under the Matern rule. "
                "Try reducing matern_radius, reducing density, or increasing max_attempts."
            )

        return Network_Law(
            name=f"Matern_radius_{matern_radius}",
            propose_edge=propose_edge,
        )

    else:
        raise ValueError(f"Unknown network law: {key}")


#############################################################################
#############################################################################


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


#############################################################################
#############################################################################

# Creates list with start and end coordinates of a unit line, given a seed position and angle.


def edge_from_seed_angle(seed, theta):
    initial_x, initial_y = seed

    end_x = initial_x + math.cos(theta)
    end_y = initial_y + math.sin(theta)

    return [[initial_x, initial_y], [end_x, end_y]]


def random_edge_uniform(L, rng=random):
    seed = [rng.uniform(0, L), rng.uniform(0, L)]
    theta = rng.uniform(0, math.pi)

    return edge_from_seed_angle(seed, theta)


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


#############################################################################
#############################################################################


def Create_pbc_Network(
    L,
    density,
    seed,
    Network_law="Uniform",
    matern_radius=0.2,
    max_attempts=10_000,
):  # positions_distribution="uniform", orientation_distribrution="uniform"):

    rng = random.Random(seed)

    network_law = make_network_law(
        Network_law,
        matern_radius=matern_radius,
        max_attempts=max_attempts,
    )

    lines = []
    nodes = []
    edges = []

    line_is_on_boundary = []
    edge_is_on_boundary = []

    N = int(5.637 * density * L**2)

    accepted_segments = initialise_segment_grid(L)

    for i in range(N):
        candidate = network_law.propose_edge(L, accepted_segments, rng)

        pbc_segments = apply_pbc(candidate[0], candidate[1], L)

        for segment in pbc_segments:
            add_segment_to_grid(accepted_segments, segment, L)

        if len(pbc_segments) > 1:
            for j in range(len(pbc_segments)):
                line_is_on_boundary.append(1)
        else:
            line_is_on_boundary.append(0)

        lines = lines + pbc_segments

    lines = [np.array(item) for item in lines]
    nodes = [item for line in lines for item in line]

    line_grid = initialise_segment_grid(L)

    for line_index, line in enumerate(lines):
        add_line_index_to_grid(line_grid, line_index, line, L)

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

        line_indices = nearby_line_indices(current_line, line_grid, L)

        for other_line_index in line_indices:
            if other_line_index <= current_line_index:
                continue

            other_line = lines[other_line_index]

            if intersect(current_line[0], current_line[1], other_line[0], other_line[1]):

                # For efficiency we loop over i,j>i, and when line i intersects line j, we record
                # that line j intersects line i in the appropriate place.

                # Find out which lines intersect each other and store pairs of indices

                intersections[current_line_index].append([current_line_index, other_line_index])
                intersections[other_line_index].append([other_line_index, current_line_index])

                # Compute coordinates of the crosslink and store it

                crosslink = intersection_line(current_line, other_line)

                crosslink_coordinates[current_line_index].append(crosslink)
                crosslink_coordinates[other_line_index].append(crosslink)

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

    # Here we create the initial incidence matrix, although it contains boundary nodes.
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
    # result in new edges and nodes being deleted, but doing so would require some form of
    # recursion, and the code would be more prone to bugs and harder to maintain.
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

    # Move boundary nodes to the end of the node list.
    # This is done after trimming, so that boundary_nodes gives the first boundary-node index.
    nodes = np.asarray(nodes)

    boundary_tol = 1e-15

    boundary_node_mask = np.any(
        (np.abs(nodes) <= boundary_tol) | (np.abs(nodes - L) <= boundary_tol),
        axis=1,
    )

    interior_node_indices = np.flatnonzero(~boundary_node_mask)
    boundary_node_indices = np.flatnonzero(boundary_node_mask)

    new_col_order = np.concatenate(
        [
            interior_node_indices,
            boundary_node_indices,
        ]
    )

    nodes = nodes[new_col_order]

    # Column slicing is faster in CSC format.
    incidence_matrix_csr = incidence_matrix.tocsc()[:, new_col_order].tocsr()

    boundary_nodes = len(interior_node_indices)

    return (
        np.array(nodes),
        boundary_nodes,
        incidence_matrix_csr,
    )


#############################################################################


def ColormapPlot_dilation(
    nodes,
    incidence_matrix,
    L,
    Lambda_1,
    Lambda_2,
    plotted_quantity,
    plotted_quantity_name="Plotted Quantity",
    density=None,
    linewidth=0.5,
    cmap=mcol.LinearSegmentedColormap.from_list(
        "network_stretch",
        [
            (0.00, "#0000FF"),  # blue
            # (0.50, "#8000FF"),  # purple
            (1.00, "#FF0000"),  # red
        ],
    ),
    robust_colour_limits=True,
    save_path=None,
):
    plotted_quantity = np.asarray(plotted_quantity)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Fast edge-node extraction.
    # This assumes each row of incidence_matrix has exactly two nonzero entries.
    edge_nodes = incidence_matrix.indices.reshape(incidence_matrix.shape[0], 2)

    segments = np.stack(
        [
            nodes[edge_nodes[:, 0]],
            nodes[edge_nodes[:, 1]],
        ],
        axis=1,
    )

    if robust_colour_limits:
        vmin, vmax = np.percentile(plotted_quantity, [1, 99])
    else:
        vmin, vmax = np.min(plotted_quantity), np.max(plotted_quantity)

    norm = mcol.Normalize(vmin=vmin, vmax=vmax)

    line_collection = LineCollection(
        segments,
        array=plotted_quantity,
        cmap=cmap,
        norm=norm,
        linewidths=linewidth,
    )

    ax.add_collection(line_collection)

    # Plot deformed domain boundary.
    ax.plot(
        [0, Lambda_1 * L, Lambda_1 * L, 0, 0],
        [0, 0, Lambda_2 * L, Lambda_2 * L, 0],
        color="black",
        linewidth=1.0,
    )

    ax.set_aspect("equal")

    ax.set_xlim(-0.1 * L, 1.1 * Lambda_1 * L)
    ax.set_ylim(-0.1 * L, 1.1 * Lambda_2 * L)

    if density is None:
        ax.set_title(r"$L = {}$".format(L))
    else:
        ax.set_title(r"$L = {}, \rho = {}$".format(L, density))

    cbar = fig.colorbar(
        line_collection,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label(plotted_quantity_name)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()

    return fig, ax


def Generate_NetworkPlot(
    L,
    density,
    seed,
    Network_law="Uniform",
    matern_radius=0.2,
    max_attempts=10000,
    linewidth=0.5,
    edge_colour="black",
    boundary_colour="black",
    save_path=None,
):
    nodes, boundary_nodes, incidence_matrix = Create_pbc_Network(
        L,
        density,
        seed,
        Network_law,
        matern_radius,
        max_attempts,
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    # Fast edge-node extraction.
    # Assumes each row of incidence_matrix has exactly two nonzero entries.
    edge_nodes = incidence_matrix.indices.reshape(incidence_matrix.shape[0], 2)

    segments = np.stack(
        [
            nodes[edge_nodes[:, 0]],
            nodes[edge_nodes[:, 1]],
        ],
        axis=1,
    )

    line_collection = LineCollection(
        segments,
        colors=edge_colour,
        linewidths=linewidth,
    )

    ax.add_collection(line_collection)

    # Domain boundary.
    ax.plot(
        [0, L, L, 0, 0],
        [0, 0, L, L, 0],
        color=boundary_colour,
        linewidth=1.0,
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.1 * L, 1.1 * L)
    ax.set_ylim(-0.1 * L, 1.1 * L)
    if Network_law == "Uniform":
        ax.set_title(r"$L = {}, \rho = {}$".format(int(L), int(density)))
    elif Network_law == "Matern":
        ax.set_title(r"$L = {}, \rho = {}, r = {}$".format(int(L), int(density), matern_radius))
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()

    return nodes, boundary_nodes, incidence_matrix


def NetworkPlot(
    L,
    nodes,
    boundary_nodes,
    incidence_matrix,
    linewidth=0.5,
    edge_colour="black",
    boundary_colour="black",
    save_path=None,
):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Fast edge-node extraction.
    # Assumes each row of incidence_matrix has exactly two nonzero entries.
    edge_nodes = incidence_matrix.indices.reshape(incidence_matrix.shape[0], 2)

    segments = np.stack(
        [
            nodes[edge_nodes[:, 0]],
            nodes[edge_nodes[:, 1]],
        ],
        axis=1,
    )

    line_collection = LineCollection(
        segments,
        colors=edge_colour,
        linewidths=linewidth,
    )

    ax.add_collection(line_collection)

    # Domain boundary.
    ax.plot(
        [0, L, L, 0, 0],
        [0, 0, L, L, 0],
        color=boundary_colour,
        linewidth=1.0,
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.1 * L, 1.1 * L)
    ax.set_ylim(-0.1 * L, 1.1 * L)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()
