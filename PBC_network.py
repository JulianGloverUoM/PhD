# -*- coding: utf-8 -*-

# Thank you Rupinder Matharu
# Code valid for integer L > 2.

import random
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from shapely.geometry import Polygon
import time

##################################################################
##################################################################
##################################################################
# Creates a random line


def random_line(L):
    # Get coordinates of a random point in the box [0, L] x [0, L]
    initial_x = random.uniform(0, L)
    initial_y = random.uniform(0, L)

    # Obtain a random angle from 0 to pi
    theta = random.uniform(0, math.pi)

    # Obtain end points of the line
    end_x = initial_x + math.cos(theta)
    end_y = initial_y + math.sin(theta)

    # Return the line coordinates
    return [[initial_x, end_x], [initial_y, end_y]]


##################################################################
##################################################################
##################################################################

# Future work, make a class for lines

# Disjoint sets function, ATM computation time scales roughly as O(N),
# with domain size having little to no affect. Optimisation possible


def disjointsets(listofsets):
    for item in listofsets:
        if type(item) != set:
            return ValueError

    count = 0
    while count < len(listofsets) - 1:
        store = []
        for item in listofsets:
            if not listofsets[count].isdisjoint(item) and listofsets[count] != item:
                store.append(item)
            else:
                continue

        if not store:
            count = count + 1

        else:
            store.append(listofsets[count])
            for item in store:
                listofsets.remove(item)
            newitem = set.union(*store)
            listofsets.append(newitem)
        continue
    return [
        set(item) for item in set(frozenset(item) for item in listofsets)
    ]  # Remove any duplicate sets


##################################################################
##################################################################
##################################################################

# Magic below
# https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


##################################################################
##################################################################
##################################################################

# Returns the intersection of 'input_line' with the box [0, L]x[0, L]


def intersection_boundary(input_line, L):
    # Check intersection with y = L
    if intersect(
        [input_line[0][0], input_line[1][0]],
        [input_line[0][1], input_line[1][1]],
        [0, L],
        [L, L],
    ):
        # Skip any divide by zero errors
        try:
            parameter = abs((L - input_line[1][0]) / (input_line[1][1] - input_line[1][0]))
            if input_line[1][0] != L:
                # Return intersection with ceiling
                return [
                    input_line[0][0] + parameter * (input_line[0][1] - input_line[0][0]),
                    L,
                ]
        except ZeroDivisionError:
            pass

    # Define N = x-coordinate of an intersection
    N = None

    # Check if intersected with wall x=0
    if intersect(
        [input_line[0][0], input_line[1][0]],
        [input_line[0][1], input_line[1][1]],
        [0, 0],
        [0, L],
    ):
        if input_line[0][0] != 0:
            N = 0
    # Check if intersected with wall x=L
    elif intersect(
        [input_line[0][0], input_line[1][0]],
        [input_line[0][1], input_line[1][1]],
        [L, 0],
        [L, L],
    ):
        if input_line[0][0] != L:
            N = L

    # If no intersection, return nothing
    if N is None:
        return

    # Skip divide by zero errors
    try:
        parameter = abs((N - input_line[0][0]) / (input_line[0][1] - input_line[0][0]))
    except ZeroDivisionError:
        return

    # Return intersection point with vertical wall
    return [N, input_line[1][0] + parameter * (input_line[1][1] - input_line[1][0])]


##################################################################
##################################################################
##################################################################

# Apply BCs to 'input_line' for the box [0, L]x[0, L] and obtain a list of lines
# corresponding to a single line crossing a box boundary


def pbc_function(input_line, L):
    # Find the intersection of 'input_line' with a box
    intersection_point = intersection_boundary(input_line, L)

    # If there is no intersection, return 'input_line' as it has no intersections
    # with the box
    if intersection_point is None:
        return [input_line]

    # Create 'first_line', the original line truncated to the intersection with the box
    first_line = [
        [input_line[0][0], intersection_point[0]],
        [input_line[1][0], intersection_point[1]],
    ]

    # Create storage for the line created after crossing the boundary of the box
    new_line = [[], []]

    # Create 'new_line'
    for i in range(2):
        if intersection_point[i] == 0 or intersection_point[i] == L:
            new_line[i].append(abs(intersection_point[i] - L))
            new_line[i].append(input_line[i][1] - intersection_point[i] + new_line[i][0])
        else:
            new_line[i].append(intersection_point[i])
            new_line[i].append(input_line[i][1])

    # Add the truncated line to a list
    output = [first_line]

    # Add all the next lines created from boundary crossings to output
    output += pbc_function(new_line, L)

    return output


##################################################################
##################################################################
##################################################################

# Finds the intersection point of two lines


def intersection_line(input_line1, input_line2):
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


##################################################################
##################################################################
##################################################################

# Intersection detection

# checks if rectangles contructed by the lines overlap


def rectangle_overlap(line1, line2):
    R1 = [
        min(line1[0][0], line1[0][1]),
        min(line1[1][0], line1[1][1]),
        max(line1[0][0], line1[0][1]),
        max(line1[1][0], line1[1][1]),
    ]

    R2 = [
        min(line2[0][0], line2[0][1]),
        min(line2[1][0], line2[1][1]),
        max(line2[0][0], line2[0][1]),
        max(line2[1][0], line2[1][1]),
    ]
    if R1[0] >= R2[2] or R1[2] <= R2[0] or R1[3] <= R2[1] or R1[1] >= R2[3]:
        return False
    else:
        return True


##################################################################
##################################################################
##################################################################

# Seperate domain into L^2 unit subcells
# Then portion line and centroid position data into respective subcells.

# Function takes in ordered list of centroid coordinates and corresponding lines.
# Note only apply this function for L>2. Cells are chosen to be unit size
# Method obtained in 'Finite-size scaling in stick percolation' (Phd papers folder)


def subcellpartition(input_centroid_coordinates, input_lines, L):
    # Data is stored in the partitioned_data list.
    # In the first layer of nesting is three LxL 'Matrices', with the [i][j]^th
    # element of the matrix containing a list of lines whose centroid
    # coordinates are contained within the subcell (i-1,i)x(j-1,j).
    # The first matrix contains the centroid coordinates.
    # The second the line_data.
    # The third the index's of these lines in original lists' ordering.
    if L <= 2:
        return print("Box size is too small for unit size subcell partition")

    partitioned_data = [[], [], []]
    for i in range(L):
        partitioned_data[0].append([])
        partitioned_data[1].append([])
        partitioned_data[2].append([])
        for j in range(L):
            partitioned_data[0][i].append([])
            partitioned_data[1][i].append([])
            partitioned_data[2][i].append([])

    for index in range(len(input_centroid_coordinates)):
        i_index = math.floor(input_centroid_coordinates[index][1])
        j_index = math.floor(input_centroid_coordinates[index][0])

        if i_index == L:
            i_index -= 1

        if j_index == L:
            j_index -= 1

        partitioned_data[0][i_index][j_index].append(input_centroid_coordinates[index])
        partitioned_data[1][i_index][j_index].append(input_lines[index])
        partitioned_data[2][i_index][j_index].append(index)

    return partitioned_data


##################################################################
##################################################################
##################################################################

# Applies intersectionsets function using data partitioned into subcells


def subcellintersectionsets(matrix, L):
    intersectionsets_list = []
    for i_index in range(L):
        for j_index in range(L):
            temp_cell_centroid = []
            temp_cell_lines = []
            temp_cell_indices = []

            for i_para in range(3):
                for j_para in range(3):
                    try:
                        temp_cell_centroid += matrix[0][i_index - 1 + i_para][j_index - 1 + j_para]
                        temp_cell_lines += matrix[1][i_index - 1 + i_para][j_index - 1 + j_para]
                        temp_cell_indices += matrix[2][i_index - 1 + i_para][j_index - 1 + j_para]
                    except IndexError:
                        pass

            loop_indices = []
            for coordinate in matrix[0][i_index][j_index]:
                loop_indices = intersectionset(
                    coordinate,
                    matrix[1][i_index][j_index][matrix[0][i_index][j_index].index(coordinate)],
                    temp_cell_centroid,
                    temp_cell_lines,
                )
                templist = [temp_cell_indices[indice] for indice in loop_indices]
                intersectionsets_list.append(templist)
    return intersectionsets_list


##################################################################
##################################################################
##################################################################


def centre(input_segment):
    return np.array([0.5 * sum(input_segment[0]), 0.5 * sum(input_segment[1])])


##################################################################
##################################################################
##################################################################


def vector_of_magnitudes(input_array):
    return np.array([np.linalg.norm(vector) for vector in input_array])


##################################################################
##################################################################
##################################################################

# generates the set of lines that intersect inputted line


def intersectionset(input_centre, input_line, input_centroid_coordinates, input_lines):
    np_centre = np.array(input_centre)

    store = [input_centroid_coordinates.index(input_centre)]
    for index in range(len(input_centroid_coordinates)):
        coordinate = np.array(input_centroid_coordinates[index])
        if (
            np.linalg.norm(np_centre - coordinate) < 1
            and rectangle_overlap(input_line, input_lines[index])
            and intersect(
                [input_line[0][0], input_line[1][0]],
                [input_line[0][1], input_line[1][1]],
                [input_lines[index][0][0], input_lines[index][1][0]],
                [input_lines[index][0][1], input_lines[index][1][1]],
            )
            and index not in store
        ):
            store.append(index)
        else:
            continue

    return store


##################################################################
##################################################################
##################################################################

# Takes in list of line segments and outputs the intersections


def intersectionset_2(input_line, input_coordinates):
    input_line_index = input_coordinates.index(input_line)
    input_line_centre = []

    output = [input_line_index]

    for i in range(len(input_line)):
        input_line_centre.append(centre(input_line[i]))

    for line in input_coordinates:
        for comparison_segment in line:
            for segment in input_line:
                if (
                    np.linalg.norm(
                        centre(comparison_segment) - input_line_centre[input_line.index(segment)]
                    )
                    <= 1
                    and rectangle_overlap(segment, comparison_segment)
                    and intersect(
                        [segment[0][0], segment[1][0]],
                        [segment[0][1], segment[1][1]],
                        [comparison_segment[0][0], comparison_segment[1][0]],
                        [comparison_segment[0][1], comparison_segment[1][1]],
                    )
                    and segment != comparison_segment
                ):
                    output.append(input_coordinates.index(line))
    return output


##################################################################
##################################################################
##################################################################


def generategraphdata(input_intersectionsets_list, input_lines):
    data = []
    for item in input_intersectionsets_list:
        if len(item) <= 2:
            continue
        temp1 = []
        for index in range(len(item) - 1):
            temp1.append(
                [
                    np.array(intersection_line(input_lines[item[0]], input_lines[item[index + 1]])),
                    item[0],
                    item[index + 1],
                ]
            )

        temp2 = []
        referencepoint = np.array([input_lines[item[0]][0][0], input_lines[item[0]][1][0]])
        # The below is a gross way of doing this,
        # my brain hurty so cant think of anything nicer atm

        for i in range(len(temp1)):
            temp2.append(np.linalg.norm(referencepoint - np.array(temp1[i][0])))
        temp3 = temp2.copy()
        temp3.sort()
        output = []
        for j in range(len(temp1)):
            output.append(
                [
                    list(temp1[temp2.index(temp3[j])][0]),
                    temp1[temp2.index(temp3[j])][1],
                    temp1[temp2.index(temp3[j])][2],
                ]
            )
        data.append(output)

    store = []
    nodes = []
    for item in data:
        for k in range(len(item)):
            item[k] = [item[k][0], set([item[k][1], item[k][2]])]

            if item[k][1] in store:
                item[k].append(store.index(item[k][1]))
            else:
                store.append(item[k][1])
                nodes.append(item[k][0])
                item[k].append(store.index(item[k][1]))

    node_connections = []
    edge_orientations = []
    for item in data:
        for k in range(len(item) - 1):
            node_connections.append(list(set((item[k][2], item[k + 1][2]))))
            edge_orientations.append(random.randint(0, 1) * 2 - 1)

    edges = []
    for item in node_connections:
        edges.append(
            [
                [nodes[item[0]][0], nodes[item[1]][0]],
                [nodes[item[0]][1], nodes[item[1]][1]],
            ]
        )

    incidence_matrix_list = []
    for i in range(len(edges)):
        incidence_matrix_list.append([])
        for j in range(len(nodes)):
            incidence_matrix_list[i].append([0])

    for index in range(len(node_connections)):
        incidence_matrix_list[index][node_connections[index][0] - 1] = [
            -1 * edge_orientations[index]
        ]
        incidence_matrix_list[index][node_connections[index][1] - 1] = [edge_orientations[index]]

    return (
        data,
        edges,
        nodes,
        node_connections,
        edge_orientations,
        incidence_matrix_list,
    )


##################################################################
##################################################################
##################################################################

# segment_intersections[item_index] contains the information about what are the
# points of intersection for each segment of line item[item_index][0]
# segment_intersections[item_index][segment_index] is specific segment


def generategraphdata_2(input_intersectionsets, input_lines):
    segment_intersections = []
    index_store = []
    for item in input_intersectionsets:
        item_index = input_intersectionsets.index(item)
        reference_line_index = item[0]
        reference_line = input_lines[reference_line_index]
        num_segments = len(reference_line)
        segment_intersections.append([])
        index_store.append([])
        for i in range(num_segments):
            segment_intersections[item_index].append(
                [[reference_line[i][0][0], reference_line[i][1][0]]]
            )
            index_store[item_index].append([tuple([reference_line_index, i])])
        segment_index = 0

        for reference_segment in reference_line:
            for i in range(len(item) - 1):
                j = i + 1
                comparison_line = input_lines[item[j]]

                for comparison_segment in comparison_line:
                    if intersect(
                        [reference_segment[0][0], reference_segment[1][0]],
                        [reference_segment[0][1], reference_segment[1][1]],
                        [comparison_segment[0][0], comparison_segment[1][0]],
                        [comparison_segment[0][1], comparison_segment[1][1]],
                    ):
                        # Adding the point of intersection to the relevant
                        # list.
                        segment_intersections[item_index][segment_index].append(
                            intersection_line(reference_segment, comparison_segment)
                        )
                        comparison_segment_index = comparison_line.index(comparison_segment)
                        # Adding the retrival data for the intersecting segments
                        index_store[item_index][segment_index].append(
                            tuple([item[j], comparison_segment_index])
                        )
            segment_index += 1

    #################################
    check = 1
    while check > 0:
        check = 0
        for line in segment_intersections:
            if len(line) == 1 and len(line[0]) <= 2:
                segment_intersections.remove(line)
                check += 1
            else:
                for segment in line:
                    if len(segment) == 1:
                        line_index = segment_intersections.index(line)
                        segment_intersections[line_index].remove(segment)
                        check += 1

    #################################
    check = 1
    while check > 0:
        check = 0
        for line in index_store:
            if len(line) == 1 and len(line[0]) <= 2:
                index_store.remove(line)
                check += 1
            else:
                for segment in line:
                    if len(segment) == 1:
                        line_index = index_store.index(line)
                        index_store[line_index].remove(segment)
                        check += 1

    #################################
    # Finding the correct ordering of intersections for each segment
    for line in segment_intersections:
        for segment in line:
            if len(segment) == 2:
                continue
            reference = np.array(segment[0])
            norms = []
            for i in range(len(segment)):
                norms.append(np.linalg.norm(np.array(segment[i]) - reference))
            sorted_norms = sorted(norms)
            indices = [norms.index(item) for item in sorted_norms]
            ordered_intersections = [segment[index] for index in indices]
            line_index = segment_intersections.index(line)
            segment_index = segment_intersections[line_index].index(segment)
            segment_intersections[line_index][segment_index] = ordered_intersections

            index_store_segment = index_store[line_index][segment_index]
            ordered_index_store = [index_store_segment[index] for index in indices]
            index_store[line_index][segment_index] = ordered_index_store

    #################################
    edges = []
    node_connections = []
    for line in segment_intersections:
        L_index = segment_intersections.index(line)
        index_data = index_store[L_index]

        # Edges when only one segement of the line intersects any other line
        if len(line) == 1:
            for i in range(len(line[0]) - 2):
                edges.append([[line[0][i + 1], line[0][i + 2]]])
                node_connections.append(
                    [
                        [index_data[0][0], index_data[0][i + 1]],
                        [index_data[0][0], index_data[0][i + 2]],
                    ]
                )

        #################################
        elif len(line) == 2:
            # The way the lines are generated, for any line our the chosen
            # reference points for each segment are set up such that for any
            # edge that cross the boundary of domain the connecting nodes are
            # stored as the last and first nodes of the coonnecting segments.
            # Case when only two segments of the line intersect other sticks
            last_node = line[0][-1]
            first_node = line[1][1]
            # Case when two segments are adjancent
            if abs(index_data[0][0][1] - index_data[1][0][1]) == 1:
                for i in range(len(line[0]) - 2):
                    edges.append([[line[0][i + 1], line[0][i + 2]]])
                    node_connections.append(
                        [
                            [index_data[0][0], index_data[0][i + 1]],
                            [index_data[0][0], index_data[0][i + 2]],
                        ]
                    )
                line_number = index_data[0][0][0]
                first_segment = index_data[0][0][1]
                first_boundary_point = [
                    input_lines[line_number][first_segment][0][1],
                    input_lines[line_number][first_segment][1][1],
                ]
                second_boundary_point = line[1][0]
                edges.append(
                    [[last_node, first_boundary_point], [second_boundary_point, first_node]]
                )
                node_connections.append(
                    [[index_data[0][0], index_data[0][-1]], [index_data[1][0], index_data[1][1]]]
                )
                for i in range(len(line[1]) - 2):
                    edges.append([[line[1][i + 1], line[1][i + 2]]])
                    node_connections.append(
                        [
                            [index_data[1][0], index_data[1][i + 1]],
                            [index_data[1][0], index_data[1][i + 2]],
                        ]
                    )

            #################################
            # Case where there is a 'non-intersected' segement involved
            # Ie a stick that crosses the boundary twice intersects some
            # lines in its first and last segments, but the segment that
            # touches the boundary twice intersects nothing
            else:
                for i in range(len(line[0]) - 2):
                    edges.append([[line[0][i + 1], line[0][i + 2]]])
                    node_connections.append(
                        [
                            [index_data[0][0], index_data[0][i + 1]],
                            [index_data[0][0], index_data[0][i + 2]],
                        ]
                    )
                line_number = index_data[0][0][0]
                first_segment = input_lines[line_number][0]
                second_segment_data = input_lines[line_number][1]
                second_segment = [
                    [second_segment_data[0][0], second_segment_data[1][0]],
                    [second_segment_data[0][1], second_segment_data[1][1]],
                ]
                first_boundary_point = [first_segment[0][1], first_segment[1][1]]
                second_boundary_point = line[1][0]
                edges.append(
                    [
                        [last_node, first_boundary_point],
                        second_segment,
                        [second_boundary_point, first_node],
                    ]
                )
                node_connections.append(
                    [[index_data[0][0], index_data[0][-1]], [index_data[1][0], index_data[1][1]]]
                )
                for i in range(len(line[1]) - 2):
                    edges.append([[line[1][i + 1], line[1][i + 2]]])
                    node_connections.append(
                        [
                            [index_data[1][0], index_data[1][i + 1]],
                            [index_data[1][0], index_data[1][i + 2]],
                        ]
                    )

        #################################
        else:
            last_node_1 = line[0][-1]
            first_node_1 = line[1][1]
            last_node_2 = line[1][-1]
            first_node_2 = line[2][1]
            for i in range(len(line[0]) - 2):
                edges.append([[line[0][i + 1], line[0][i + 2]]])
                node_connections.append(
                    [
                        [index_data[0][0], index_data[0][i + 1]],
                        [index_data[0][0], index_data[0][i + 2]],
                    ]
                )
            line_number = index_data[0][0][0]
            first_boundary_point = [
                input_lines[line_number][0][0][1],
                input_lines[line_number][0][1][1],
            ]
            second_boundary_point = line[1][0]
            edges.append(
                [[last_node_1, first_boundary_point], [second_boundary_point, first_node_1]]
            )
            node_connections.append(
                [[index_data[0][0], index_data[0][-1]], [index_data[1][0], index_data[1][1]]]
            )
            for i in range(len(line[1]) - 2):
                edges.append([[line[1][i + 1], line[1][i + 2]]])
                node_connections.append(
                    [
                        [index_data[1][0], index_data[1][i + 1]],
                        [index_data[1][0], index_data[1][i + 2]],
                    ]
                )
            third_boundary_point = [
                input_lines[line_number][1][0][1],
                input_lines[line_number][1][1][1],
            ]
            fourth_boundary_point = line[2][0]
            edges.append(
                [[last_node_2, third_boundary_point], [fourth_boundary_point, first_node_2]]
            )
            node_connections.append(
                [[index_data[1][0], index_data[1][-1]], [index_data[2][0], index_data[2][1]]]
            )
            for i in range(len(line[2]) - 2):
                edges.append([[line[2][i + 1], line[2][i + 2]]])
                node_connections.append(
                    [
                        [index_data[2][0], index_data[2][i + 1]],
                        [index_data[2][0], index_data[2][i + 2]],
                    ]
                )

    #################################
    store = []
    data = []
    # Currently grows exponentially, but still dominated by matrix trimming
    for line in index_store:
        for segment in line:
            for i in range(len(segment) - 1):
                if not set([segment[i], segment[i + 1]]) in store:
                    store.append(set([segment[0], segment[i + 1]]))
                    line_index = index_store.index(line)
                    segment_index = line.index(segment)
                    intersection_coordinate = segment_intersections[line_index][segment_index][
                        i + 1
                    ]
                    data.append(
                        [
                            [intersection_coordinate],
                            set([segment[0], segment[i + 1]]),
                            store.index(set([segment[0], segment[i + 1]])),
                        ]
                    )

    #################################
    # This is currently expensive for dense networks. needs some work

    # start_time = time.time()
    edge_orientations = []
    for item in node_connections:
        edge_orientations.append(random.randint(0, 1) * 2 - 1)
        for node in item:
            if set(node) in store:
                node_connections[node_connections.index(item)][item.index(node)] = store.index(
                    set(node)
                )
            else:
                store.append(set(node))
                node_connections[node_connections.index(item)][item.index(node)] = store.index(
                    set(node)
                )
    # print("time taken to reindex nodes = ", time.time() - start_time)
    #################################

    num_nodes = (
        max([max(item) for item in node_connections]) + 1
    )  # This is not right but is dealt with in trim
    num_edges = len(edges)
    incidence_matrix_EV_list = []
    for i in range(num_edges):
        incidence_matrix_EV_list.append([])
        for j in range(num_nodes):
            incidence_matrix_EV_list[i].append(0)

    for index in range(len(node_connections)):
        incidence_matrix_EV_list[index][node_connections[index][0] - 1] = (
            -1 * edge_orientations[index]
        )

        incidence_matrix_EV_list[index][node_connections[index][1] - 1] = edge_orientations[index]

    #################################
    # Trimming
    # start_time = time.time()

    matrix = np.array(incidence_matrix_EV_list)
    trimmed_matrix = matrix.copy()
    dangly_edges = []
    deleted_nodes = []
    counter = 1
    check = 0
    total_deleted = 0
    while True:
        check = 0
        if counter >= 1000:
            break
        j = 0
        num_deleted = 0
        iterated_matrix = trimmed_matrix.copy()
        for j in range(np.shape(trimmed_matrix)[1]):
            num_connections = len(np.nonzero(trimmed_matrix[:, j])[0])
            if num_connections > 1:
                continue
            else:
                iterated_matrix = np.delete(iterated_matrix, j - num_deleted, axis=1)
                deleted_nodes.append(j + total_deleted)
                total_deleted += 1
                num_deleted += 1
                check = 1

        for i in range(np.shape(iterated_matrix)[0]):
            if len(np.nonzero(iterated_matrix[i, :])[0]) <= 1:
                iterated_matrix[i, :] = 0
                check = 1

        trimmed_matrix = iterated_matrix.copy()
        if check == 0:
            break
        counter += 1

    # print(
    #     "time taken to generate trimmed_matrix = ",
    #     time.time() - start_time
    # )

    deleted_rows = []
    for i in range(np.shape(trimmed_matrix)[0]):
        if len(np.nonzero(trimmed_matrix[i, :])[0]) <= 1:
            dangly_edges.append(edges[i])
            deleted_rows.append(i)

    trimmed_edges = edges.copy()
    for edge in edges:
        if edge in dangly_edges:
            trimmed_edges.remove(edge)

    trimmed_matrix = np.delete(trimmed_matrix, deleted_rows, axis=0)

    nodes = []
    for j in range(np.shape(trimmed_matrix)[1]):
        (e1, e2) = (
            trimmed_edges[np.nonzero(trimmed_matrix[:, j])[0][0]],
            trimmed_edges[np.nonzero(trimmed_matrix[:, j])[0][1]],
        )
        norm = 1
        for segment in e1:
            for coord in segment:
                c1 = np.array(coord)
                for segments in e2:
                    for coords in segments:
                        c2 = np.array(coords)
                        if np.linalg.norm(c1 - c2) < norm:
                            norm = np.linalg.norm(c1 - c2)
                            node = c1
        nodes.append(np.array(node))

    return (
        data,
        segment_intersections,
        index_store,
        nodes,
        edges,
        node_connections,
        edge_orientations,
        matrix,
        trimmed_matrix,
        trimmed_edges,
    )


##################################################################
##################################################################
##################################################################
# Running the code

# N = number of lines, L = length of the box (integer)


def CreateNetwork(N, L, seed):
    random.seed(seed)
    # Storage for coordinates
    coordinates = []

    # Create N lines and apply BCs
    for i in range(N):
        line = random_line(L)
        coordinates.append(pbc_function(line, L))

    # Centroid coordinates

    coordinates_centre = []

    for line in coordinates:
        coordinates_centre.append([])
        index = coordinates.index(line)
        for segment in line:
            coordinates_centre[index].append(centre(segment))

    intersectionsets = []

    for coordinate in coordinates:
        coordinate_set = intersectionset_2(coordinate, coordinates)
        if len(coordinate_set) > 1:
            intersectionsets.append(coordinate_set)

    disjoint_intersectionsets = disjointsets([set(item) for item in intersectionsets])

    # Removal of sticks that cannot carry load

    # 1) Remove any sticks that just hang in space.
    valid_intersections_1 = []

    for item in intersectionsets:
        if set(item) in disjoint_intersectionsets and len(item) <= 2:
            continue
        valid_intersections_1.append(set(item))

    # 2) Remove clusters that do not cross the boundary

    valid_intersections_2 = [list(item) for item in disjointsets(valid_intersections_1.copy())]

    valid_intersections_3 = []

    for item in valid_intersections_2:
        item_list = list(item)
        for line in item_list:
            if len(coordinates[line]) > 1:
                valid_intersections_3.append(set(item))
                break

    valid_intersections = []
    for item in intersectionsets:
        for cluster in valid_intersections_3:
            if not set(item).isdisjoint(cluster):
                valid_intersections.append(item)

    graphdata = generategraphdata_2(valid_intersections, coordinates)

    (
        data,
        segment_intersections,
        index_store,
        nodes,
        edges,
        node_connections,
        edge_orientations,
        matrix,
        trimmed_matrix,
        trimmed_edges,
    ) = graphdata

    return (
        [coordinates, edges, trimmed_edges],
        [intersectionsets, valid_intersections, nodes],
        [matrix, trimmed_matrix],
    )


# (coordinates, edges, trimmed_edges) = CreateNetwork(N, L)[0]
# (intersectionsets, valid_intersections) = CreateNetwork(N, L)[1]
# (matrix, trimmed_matrix) = CreateNetwork(N, L)[2]

##################################################################
##################################################################
##################################################################
# Tests

# Histogram of orientations

# Length in the box

# Subcell and full domain methods generate the same intersectionsets

# unfinished atm


def TestSubcellIntersectionsetsDisjoint(input_intersectionsets, input_intersectionsets_subcell):
    if len(input_intersectionsets) != len(input_intersectionsets_subcell):
        return False
    else:
        return True
    return


# This is O(n^2) expensive. Do not run unless neccessary
# Generates disjoint intersectionsets of the lines using naive method to
# determine intersections.


def TestIntersections(input_line_data, input_disjoint_intersectionsets):
    output = []
    for tested_line in input_line_data:
        output.append([input_line_data.index(tested_line)])
        for comparison_line in input_line_data:
            if comparison_line == tested_line:
                continue
            intersection_coordinates = intersection_line(tested_line, comparison_line)
            if not intersection_coordinates:
                continue
            x_min_tested = min(tested_line[0])
            x_max_tested = max(tested_line[0])

            x_min_comparison = min(comparison_line[0])
            x_max_comparison = max(comparison_line[0])
            if (
                x_min_tested <= intersection_coordinates[0] <= x_max_tested
                and x_min_comparison <= intersection_coordinates[0] <= x_max_comparison
            ):
                output[input_line_data.index(tested_line)].append(
                    input_line_data.index(comparison_line)
                )
    Naive_disjointsets = disjointsets([set(item) for item in output])
    for item in Naive_disjointsets:
        check = item in input_disjoint_intersectionsets
        if check is False:
            return False
    return True


# Checks that applying the PBC does not create or lose any length in the system.
# Returns the deviation from the 'corrrect' length.
# Floating point error does not accumulate for N <= 300, unsure for larger N.


def TestStickLengths(input_sticks, num_sticks):
    total_length = 0
    for line in input_sticks:
        line_length = 0
        for segment in line:
            first_coord = np.array([segment[0][0], segment[1][0]])
            second_coord = np.array([segment[0][1], segment[1][1]])
            segment_length = np.linalg.norm(first_coord - second_coord)
            line_length += segment_length
        if line_length > 1:
            print(input_sticks.index(line))
            print(line_length)
        total_length += line_length

    if total_length != num_sticks:
        print("Total length of sticks=", total_length)
        print("Deviation from correct length =", total_length - num_sticks)

    return total_length


# Check edges are not too long


def TestEdgeLengths(input_edges):
    norms = []
    for edge in input_edges:
        norms.append([])
        edge_index = input_edges.index(edge)
        for segment in edge:
            first_coord = np.array(segment[0])
            second_coord = np.array(segment[1])
            norm = np.linalg.norm(first_coord - second_coord)
            norms[edge_index].append(norm)
            if norm > 1:
                print(segment)
                print(edge_index, edge.index(segment))


##################################################################
##################################################################
##################################################################
# Plotting

# Plotting Sticks

# Loop through all the lines and ensure lines that occur after crossing a
# boundary are the same colour as the line they come from


def Plotfigures(coordinates, edges, trimmed_edges, L):
    plt.subplot(1, 3, 1)

    for line in coordinates:
        # Set the colour for the next line (I hate the American spelling)
        color = next(plt.gca()._get_lines.prop_cycler)["color"]
        for segment in line:
            plt.plot(segment[0], segment[1], color=color, linewidth=1)

    # Plot the box
    plt.plot([0, L, L, 0, 0], [0, 0, L, L, 0])

    plt.xlim(0 - 0.1 * L, 1.1 * L)
    plt.ylim(0 - 0.1 * L, 1.1 * L)

    plt.gca().set_aspect("equal")
    # plt.savefig('Debug_G1.pdf')

    # Plotting graph

    plt.subplot(1, 3, 2)

    for edge in edges:
        color = next(plt.gca()._get_lines.prop_cycler)["color"]
        for segment in edge:
            plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color=color)

    # Plot the box
    plt.plot([0, L, L, 0, 0], [0, 0, L, L, 0])

    # Set the limits to slightly larger than the box
    plt.xlim(0 - 0.1 * L, 1.1 * L)
    plt.ylim(0 - 0.1 * L, 1.1 * L)

    plt.gca().set_aspect("equal")
    # plt.savefig('Debug_G2.pdf')

    plt.subplot(1, 3, 3)

    for edge in trimmed_edges:
        color = next(plt.gca()._get_lines.prop_cycler)["color"]
        for segment in edge:
            plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color=color)

    # Plot the box
    plt.plot([0, L, L, 0, 0], [0, 0, L, L, 0])

    # Set the limits to slightly larger than the box
    plt.xlim(0 - 0.1 * L, 1.1 * L)
    plt.ylim(0 - 0.1 * L, 1.1 * L)

    plt.gca().set_aspect("equal")
    # plt.savefig('Debug_G3.pdf')
    return


def PlotNetwork(edges, L):
    fig_network = plt.figure()
    for edge in edges:
        # color = next(plt.gca()._get_lines.prop_cycler)["color"]
        for segment in edge:
            plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color="b")

    # Plot the box
    plt.plot([0, L, L, 0, 0], [0, 0, L, L, 0])

    plt.xlim(0 - 0.1 * L, 1.1 * L)
    plt.ylim(0 - 0.1 * L, 1.1 * L)

    plt.gca().set_aspect("equal")
    plt.show
    return


def PlotNetwork_2(nodes, incidence_matrix, L, shear_factor):
    new_edges = []
    for row in incidence_matrix:
        node_1 = list(nodes[np.nonzero(row)[0][0]])
        node_2 = list(nodes[np.nonzero(row)[0][1]])
        new_edges.append([node_1, node_2])

    fig_final = plt.figure()

    for edge in new_edges:
        plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color="b")

    plt.plot([0, L, (1 + shear_factor) * L, shear_factor * L, 0], [0, 0, L, L, 0])

    plt.xlim(0 - 0.1 * L, 1.1 * (1 + shear_factor) * L)
    plt.ylim(0 - 0.1 * L, 1.1 * L)

    plt.gca().set_aspect("equal")

    # plt.title(str(r"$\gamma$ = {}".format(shear_factor)))
    plt.savefig("generated_network_example_2.pdf")
    return


def ColormapPlot(nodes, incidence_matrix, L, shear_factor, initial_lengths):
    strains = (
        vector_of_magnitudes(incidence_matrix.dot(nodes)) - initial_lengths
    ) / initial_lengths

    cm1 = mcol.LinearSegmentedColormap.from_list("bpr", ["w", "r"])
    cnorm = mcol.Normalize(vmin=min(strains), vmax=max(strains))
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])
    fig = plt.figure()
    plt.title(str(r"$\gamma$ = {}".format(shear_factor)))
    plt.gca().set_aspect("equal")
    new_edges = []
    for row in incidence_matrix:
        node_1 = list(nodes[np.nonzero(row)[0][0]])
        node_2 = list(nodes[np.nonzero(row)[0][1]])
        new_edges.append([node_1, node_2])

    for i in range(len(new_edges)):
        edge = new_edges[i]

        plt.plot(
            [edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color=cpick.to_rgba(strains[i])
        )

    plt.plot([0, L, (1 + shear_factor) * L, shear_factor * L, 0], [0, 0, L, L, 0])
    ax = plt.gca()

    plt.xlim(0 - 0.1 * L, 1.1 * (1 + shear_factor) * L)
    plt.ylim(0 - 0.1 * L, 1.1 * L)

    plt.colorbar(
        cpick,
        cax=fig.add_axes([0.85, 0.25, 0.05, 0.5]),
        boundaries=np.arange(min(strains), max(strains), (max(strains) - min(strains)) / 100),
    )

    return


def ColormapPlot_stretch(nodes, incidence_matrix, L, stretch_factor, initial_lengths):
    strains = (
        vector_of_magnitudes(incidence_matrix.dot(nodes)) - initial_lengths
    ) / initial_lengths

    cm1 = mcol.LinearSegmentedColormap.from_list("bpr", ["w", "r"])
    cnorm = mcol.Normalize(vmin=min(strains), vmax=max(strains))
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])
    fig = plt.figure()
    plt.title(str(r"$\lambda$ = {}".format(stretch_factor)))
    plt.gca().set_aspect("equal")
    new_edges = []
    for row in incidence_matrix:
        node_1 = list(nodes[np.nonzero(row)[0][0]])
        node_2 = list(nodes[np.nonzero(row)[0][1]])
        new_edges.append([node_1, node_2])

    for i in range(len(new_edges)):
        edge = new_edges[i]

        plt.plot(
            [edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color=cpick.to_rgba(strains[i])
        )

    plt.plot(
        [0, (stretch_factor) * L, (stretch_factor) * L, 0, 0],
        [0, 0, (stretch_factor) * L, (stretch_factor) * L, 0],
    )
    ax = plt.gca()

    plt.xlim(0 - 0.1 * L, 1.1 * (stretch_factor) * L)
    plt.ylim(0 - 0.1 * L, 1.1 * (stretch_factor) * L)

    plt.colorbar(
        cpick,
        cax=fig.add_axes([0.85, 0.25, 0.05, 0.5]),
        boundaries=np.arange(min(strains), max(strains), (max(strains) - min(strains)) / 100),
    )

    return


def ColormapPlot_uniaxial_stretch(nodes, incidence_matrix, L, stretch_factor, initial_lengths):
    strains = (
        vector_of_magnitudes(incidence_matrix.dot(nodes)) - initial_lengths
    ) / initial_lengths

    cm1 = mcol.LinearSegmentedColormap.from_list("bpr", ["w", "r"])
    cnorm = mcol.Normalize(vmin=min(strains), vmax=max(strains))
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])
    fig = plt.figure()
    plt.title(str(r"$\lambda$ = {}".format(stretch_factor)))
    # plt.gca().set_aspect("equal")
    new_edges = []
    for row in incidence_matrix:
        node_1 = list(nodes[np.nonzero(row)[0][0]])
        node_2 = list(nodes[np.nonzero(row)[0][1]])
        new_edges.append([node_1, node_2])

    for i in range(len(new_edges)):
        edge = new_edges[i]

        plt.plot(
            [edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color=cpick.to_rgba(strains[i])
        )

    plt.plot(
        [0, L, L, 0, 0],
        [0, 0, (1 + stretch_factor) * L, (1 + stretch_factor) * L, 0],
    )
    ax = plt.gca()

    plt.xlim(0 - 0.1 * L, 1.1 * L)
    plt.ylim(0 - 0.1 * L, 1.1 * (stretch_factor) * L)

    plt.colorbar(
        cpick,
        cax=fig.add_axes([0.85, 0.25, 0.05, 0.5]),
        boundaries=np.arange(min(strains), max(strains), (max(strains) - min(strains)) / 100),
    )

    return


def ColormapPlot_dilation(nodes, incidence_matrix, L, lambda_1, lambda_2, initial_lengths):
    strains = (
        vector_of_magnitudes(incidence_matrix.dot(nodes)) - initial_lengths
    ) / initial_lengths

    cm1 = mcol.LinearSegmentedColormap.from_list("bpr", ["b", "r"])
    cnorm = mcol.Normalize(vmin=min(strains), vmax=max(strains))
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])
    fig = plt.figure()
    plt.title(str(r"$\lambda_1,\lambda_2$ = {},{}".format(lambda_1, lambda_2)))
    plt.gca().set_aspect("equal")

    new_edges = []
    for row in incidence_matrix:
        node_1 = list(nodes[np.nonzero(row)[0][0]])
        node_2 = list(nodes[np.nonzero(row)[0][1]])
        new_edges.append([node_1, node_2])

    for i in range(len(new_edges)):
        edge = new_edges[i]

        plt.plot(
            [edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color=cpick.to_rgba(strains[i])
        )

    plt.plot(
        [0, lambda_1 * L, lambda_1 * L, 0, 0],
        [0, 0, lambda_2 * L, lambda_2 * L, 0],
    )
    ax = plt.gca()

    plt.xlim(0 - 0.1 * L, 1.1 * lambda_1 * L)
    plt.ylim(0 - 0.1 * L, 1.1 * lambda_2 * L)

    plt.colorbar(
        cpick,
        cax=fig.add_axes([0.85, 0.25, 0.05, 0.5]),
        boundaries=np.arange(min(strains), max(strains), (max(strains) - min(strains)) / 100),
    )
    plt.savefig("prestress_network_deformed_equilibrium_example.pdf")

    return
