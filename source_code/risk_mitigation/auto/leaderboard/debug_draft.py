import numpy as np
import copy
from argoverse.utils import interpolate


def calculate_outreach_from_normal_left_handed():
    # lane_wp1 = wp
    # lane_wp1_loc = np.array([
    #     lane_wp1.transform.location.x,
    #     lane_wp1.transform.location.y
    # ])
    # lane_wp2 = wp.next(0.1)
    # assert len(lane_wp2) == 1, "Error: Multiple options available."
    # lane_wp2 = lane_wp2[0]
    # lane_wp2_loc = np.array([
    #     lane_wp2.transform.location.x,
    #     lane_wp2.transform.location.y
    # ])
    # lane_width = (lane_wp1.lane_width + lane_wp2.lane_width) / 2
    # center_points = np.array([lane_wp1_loc, lane_wp2_loc])
    # dense_lane_vec = interpolate.interp_arc(10, center_points[:, 0], center_points[:, 1])
    lane_width = 3.5
    center_points = np.array([[0, 0],
                              [-1, -2]])
    dense_lane_vec = interpolate.interp_arc(10, center_points[:, 0], center_points[:, 1])
    lane_wp1_loc = center_points[0]
    avgDx = np.average(np.gradient(dense_lane_vec[:, 0]))
    avgDy = np.average(np.gradient(dense_lane_vec[:, 1]))

    # handle special case with vertical slopes
    if avgDx == 0:
        print("Encounter special case in the outreach function, vertical motion.")
        # travelling downwards
        if dense_lane_vec[0, 1] > dense_lane_vec[-1, 1]:
            leftSide = [lane_wp1_loc[0] + lane_width / 2.0, lane_wp1_loc[1]]
            rightSide = [lane_wp1_loc[0] - lane_width / 2.0, lane_wp1_loc[1]]
            leftSideBound = [lane_wp1_loc[0] + lane_width / 2.0 + 0.1, lane_wp1_loc[1]]
            rightSideBound = [lane_wp1_loc[0] - lane_width / 2.0 - 0.1, lane_wp1_loc[1]]
        # travelling upwards
        elif dense_lane_vec[0, 1] < dense_lane_vec[-1, 1]:
            leftSide = [lane_wp1_loc[0] - lane_width / 2.0, lane_wp1_loc[1]]
            rightSide = [lane_wp1_loc[0] + lane_width / 2.0, lane_wp1_loc[1]]
            leftSideBound = [lane_wp1_loc[0] - lane_width / 2.0 - 0.1, lane_wp1_loc[1]]
            rightSideBound = [lane_wp1_loc[0] + lane_width / 2.0 + 0.1, lane_wp1_loc[1]]
        else:
            raise Exception("The waypoints must move.")
        return leftSide, rightSide, leftSideBound, rightSideBound

    # calculate slope
    slope = avgDy / avgDx

    if slope == 0:
        print("Encounter special case in the outreach function, horizontal motion.")
        # travelling left
        if dense_lane_vec[0, 0] > dense_lane_vec[-1, 0]:
            leftSide = [lane_wp1_loc[0], lane_wp1_loc[1] - lane_width / 2.0]
            rightSide = [lane_wp1_loc[0], lane_wp1_loc[1] + lane_width / 2.0]
            leftSideBound = [lane_wp1_loc[0], lane_wp1_loc[1] - lane_width / 2.0 - 0.1]
            rightSideBound = [lane_wp1_loc[0], lane_wp1_loc[1] + lane_width / 2.0 + 0.1]
        # travelling right
        elif dense_lane_vec[0, 0] < dense_lane_vec[-1, 0]:
            leftSide = [lane_wp1_loc[0], lane_wp1_loc[1] + lane_width / 2.0]
            rightSide = [lane_wp1_loc[0], lane_wp1_loc[1] - lane_width / 2.0]
            leftSideBound = [lane_wp1_loc[0], lane_wp1_loc[1] + lane_width / 2.0 + 0.1]
            rightSideBound = [lane_wp1_loc[0], lane_wp1_loc[1] - lane_width / 2.0 - 0.1]
        else:
            raise Exception("The waypoints must move.")
        return leftSide, rightSide, leftSideBound, rightSideBound

    # calculate inverse slope and angle
    invSlope = -1.0 / slope
    theta = np.arctan(invSlope)
    xDiff = lane_width / 2.0 * np.cos(theta)
    yDiff = lane_width / 2.0 * np.sin(theta)
    xBoundDiff = 0.1 * np.cos(theta)
    yBoundDiff = 0.1 * np.sin(theta)

    # the operation for X axis is reversed because in Unreal engine we have an inverted (left-handed) coord system
    leftSide = [lane_wp1_loc[0] - xDiff, lane_wp1_loc[1] - yDiff]
    rightSide = [lane_wp1_loc[0] + xDiff, lane_wp1_loc[1] + yDiff]
    leftSideBound = [lane_wp1_loc[0] - xDiff - xBoundDiff, lane_wp1_loc[1] - yDiff - yBoundDiff]
    rightSideBound = [lane_wp1_loc[0] + xDiff + xBoundDiff, lane_wp1_loc[1] + yDiff + yBoundDiff]

    if (avgDx > 0 and avgDy < 0) or (avgDx > 0 and avgDy > 0):
        tempSide = copy.deepcopy(leftSide)
        leftSide = rightSide
        rightSide = tempSide
        tempSideBound = copy.deepcopy(leftSideBound)
        leftSideBound = rightSideBound
        rightSideBound = tempSideBound
    if yDiff < 0:
        tempSide = copy.deepcopy(leftSide)
        leftSide = rightSide
        rightSide = tempSide
        tempSideBound = copy.deepcopy(leftSideBound)
        leftSideBound = rightSideBound
        rightSideBound = tempSideBound
    return leftSide, rightSide, leftSideBound, rightSideBound


if __name__ == "__main__":
    print(calculate_outreach_from_normal_left_handed())
