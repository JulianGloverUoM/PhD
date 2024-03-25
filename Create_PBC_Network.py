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
        [(item - L) <= 1e-15 for item in boundary_node_1]
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


def Createnetwork(
    L,
    density,
    seed,
):  # positions_distribution="uniform", orientation_distribrution="uniform"):
    random.seed(seed)

    lines = []
    for i in range(int(density * L**2)):  # generate a bunch of line segements under PBC
        line = random_edge_uniform(L)
        line = apply_pbc(line[0], line[1], L)
        lines = lines + line
    lines = [np.array(item) for item in lines]

    intersections = []
    crosslink_coordinates = []

    for (reference_line_index, reference_line) in enumerate(lines):
        other_line_index = reference_line_index + 1
        # Run intersection check over other elements of the list ignoring duplicates
        for (line_index, other_line) in enumerate(lines[reference_line_index + 1 :]):
            if intersect(reference_line[0], reference_line[1], other_line[0], other_line[1]):
                other_line_index = line_index + reference_line_index + 1
                intersections.append([reference_line_index, other_line_index])
                crosslink_coordinates.append(intersection_line(reference_line, other_line))
                # FIGURE OUT THE ORDER HERE!

    # boundary_nodes = []
    # boundary_indices = []
    # for (i, entry) in enumerate(lines):
    #     for (j, element) in enumerate(entry):
    #         if any(np.mod(element, L) <= 1e-15):
    #             boundary_nodes.append(entry)
    #             boundary_indices.append([i, j])

    nodes_list = []  # We now create a list of all the nodes
    for item in lines:
        nodes_list.append(item[0])
        nodes_list.append(item[1])

    for item in crosslink_coordinates:
        nodes_list.append(item)

    intersection_sets = []
    for (index, item) in enumerate(intersections):
        for element in item:
            intersection_sets.append(set([2 * element, index + 2 * len(lines)]))
            intersection_sets.append(set([2 * element + 1, index + 2 * len(lines)]))
    return (lines, intersections, crosslink_coordinates)
