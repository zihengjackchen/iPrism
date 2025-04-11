"""
MIT License

Copyright (c) 2022 Shengkun Cui, Saurabh Jha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import copy
import os.path
import sys

import carla

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from shapely import geometry
from utils.route_manipulation import interpolate_trajectory_wp
from argoverse.utils import interpolate
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from shapely.affinity import translate, rotate
from consts import *


class DataCollector:
    def __init__(self, data_save_path, data_save_name, world, map, visualize=False):
        self.data_save_path = data_save_path
        self.data_save_name = data_save_name
        print("Save to {}".format(os.path.join(self.data_save_path, data_save_name)))
        self.world = world
        self.map = map
        self.visualize = visualize
        self.rasterized_optimized_lanes_around_traj = dict()
        self.snapshot_obstacles = {"dynamicObstacles": {}, "staticObstacles": []}
        self.timestamp = list()
        self.ego_telemetry = dict()

    @staticmethod
    def calculate_outreach_from_normal(wp):
        lane_wp1 = wp
        lane_wp1_loc = np.array([
            -lane_wp1.transform.location.x,
            lane_wp1.transform.location.y
        ])
        lane_wp2 = wp.next(0.1)
        if len(lane_wp2) > 1:
            print("Warning, more than one successor is available, choose the closest first one by default.")
            min_dist = sys.float_info.max
            min_dist_wp = lane_wp2[0]
            for twp in lane_wp2:
                distance = twp.transform.location.distance(lane_wp1.transform.location)
                if distance < min_dist:
                    min_dist = distance
                    min_dist_wp = twp
            lane_wp2 = min_dist_wp
        else:
            lane_wp2 = lane_wp2[0]
        lane_wp2_loc = np.array([
            -lane_wp2.transform.location.x,
            lane_wp2.transform.location.y
        ])
        lane_width = (lane_wp1.lane_width + lane_wp2.lane_width) / 2
        center_points = np.array([lane_wp1_loc, lane_wp2_loc])
        dense_lane_vec = interpolate.interp_arc(10, center_points[:, 0], center_points[:, 1])
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

        # handle special case with horizontal slopes
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

        # calculate outreaches
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

    def append_ts(self, timestamp):
        self.timestamp.append(timestamp)

    def collect_lane_information(self, config, travel_dist):
        # 0. get all the waypoints, this is one big lane we are building
        _, _, waypoints = interpolate_trajectory_wp(self.world, config.trajectory, hop_resolution=travel_dist)
        waypoints.append(self.map.get_waypoint(config.trajectory[-1],
                                               project_to_road=True,
                                               lane_type=carla.LaneType.Driving))
        clean_waypoints = list()
        exclude_list = list()
        for idx in range(len(waypoints) - 1):
            wp1 = waypoints[idx]
            wp2 = waypoints[idx + 1]
            if wp1.transform.location.distance(wp2.transform.location) < 0.5 * travel_dist:
                exclude_list.append(idx + 1)
        print("Exclude list:", exclude_list)
        for idx in range(len(waypoints)):
            if idx not in exclude_list:
                wp1 = waypoints[idx]
                clean_waypoints.append(wp1)
        waypoints = clean_waypoints
        rasterized_optimized_lanes_around_traj = dict()
        waypoints_np = list()

        def _same_direction(wpa, wpb):
            wpa_np = np.array([
                -wpa.transform.location.x,
                wpa.transform.location.y
            ])
            if not len(wpa.next(0.1)):
                return False
            wpa_next = wpa.next(0.1)[0]
            wpa_next_np = np.array([
                -wpa_next.transform.location.x,
                wpa_next.transform.location.y
            ])
            wpa_vec = wpa_next_np - wpa_np

            wpb_np = np.array([
                -wpb.transform.location.x,
                wpb.transform.location.y
            ])
            if not len(wpb.next(0.1)):
                return False
            wpb_next = wpb.next(0.1)[0]
            wpb_next_np = np.array([
                -wpb_next.transform.location.x,
                wpb_next.transform.location.y
            ])
            wpb_vec = wpb_next_np - wpb_np
            unit_vector_1 = wpa_vec / np.linalg.norm(wpa_vec)
            unit_vector_2 = wpb_vec / np.linalg.norm(wpb_vec)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = np.arccos(dot_product)
            if angle > 0.174533 * 3:
                return False
            return True

        # 1. get the list of POI for the lane used the waypoints and include all lane change options
        lane_poi = list()
        lane_prev_poi = list()
        lane_next_poi = list()
        poi_wp_ids = set()
        for idx in range(len(waypoints) - 1):
            wp1 = waypoints[idx]
            wp1 = np.array([
                wp1.transform.location.x,
                wp1.transform.location.y,
                wp1.transform.location.z
            ])
            wp1_inv = np.array([
                -waypoints[idx].transform.location.x,
                waypoints[idx].transform.location.y,
                waypoints[idx].transform.location.z
            ])
            wp2 = waypoints[idx + 1]
            wp2 = np.array([
                wp2.transform.location.x,
                wp2.transform.location.y,
                wp2.transform.location.z
            ])
            wp2_inv = np.array([
               -waypoints[idx + 1].transform.location.x,
                waypoints[idx + 1].transform.location.y,
                waypoints[idx + 1].transform.location.z
            ])
            mwp = (wp1 + wp2) / 2  # take the middle point as point of interest
            mwp_loc = carla.Location(x=mwp[0], y=mwp[1], z=mwp[2])
            mwp_projected = self.map.get_waypoint(mwp_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

            # search left and right additional options, first search left
            lane_poi.append(np.array([
                -mwp_projected.transform.location.x,
                mwp_projected.transform.location.y,
                mwp_projected.transform.location.z
            ]))
            loc_projected_prev = mwp_projected.previous(0.1)[0]
            poi_prev = np.array([
                -loc_projected_prev.transform.location.x,
                loc_projected_prev.transform.location.y,
                loc_projected_prev.transform.location.z
            ])
            loc_projected_next = mwp_projected.next(0.1)[0]
            poi_next = np.array([
                -loc_projected_next.transform.location.x,
                loc_projected_next.transform.location.y,
                loc_projected_next.transform.location.z
            ])
            lane_prev_poi.append(poi_prev)
            lane_next_poi.append(poi_next)

            lane_change_option = str(mwp_projected.lane_change)
            left_most_wp = mwp_projected
            while lane_change_option != 'NONE':
                if lane_change_option == "Both" or lane_change_option == "Left":
                    left_most_wp = left_most_wp.get_left_lane()
                    if left_most_wp is not None:
                        lane_change_option = str(left_most_wp.lane_change)
                    else:
                        print("There is no left lane the but the key of the previous lane is:", lane_change_option)
                        break
                    if not str(left_most_wp.id) in poi_wp_ids and _same_direction(left_most_wp, mwp_projected):
                        lane_poi.append(np.array([
                            -left_most_wp.transform.location.x,
                            left_most_wp.transform.location.y,
                            left_most_wp.transform.location.z
                        ]))
                        lloc_projected_prev = left_most_wp.previous(0.1)[0]
                        poi_prev = np.array([
                            -lloc_projected_prev.transform.location.x,
                            lloc_projected_prev.transform.location.y,
                            lloc_projected_prev.transform.location.z
                        ])
                        lloc_projected_next = left_most_wp.next(0.1)[0]
                        poi_next = np.array([
                            -lloc_projected_next.transform.location.x,
                            lloc_projected_next.transform.location.y,
                            lloc_projected_next.transform.location.z
                        ])
                        lane_prev_poi.append(poi_prev)
                        lane_next_poi.append(poi_next)
                        poi_wp_ids.add(str(left_most_wp.id))
                    else:
                        print(str(left_most_wp.id), "already in the set.")
                        break  # we hit a loop here
                elif lane_change_option == "Right":
                    break

            # then search right
            lane_change_option = str(mwp_projected.lane_change)
            right_most_wp = mwp_projected
            while lane_change_option != 'NONE':
                if lane_change_option == "Both" or lane_change_option == "Right":
                    right_most_wp = right_most_wp.get_right_lane()
                    if right_most_wp is not None:
                        lane_change_option = str(right_most_wp.lane_change)
                    else:
                        print("There is no right lane the but the key of the previous lane is:", lane_change_option)
                        break
                    if not str(right_most_wp.id) in poi_wp_ids and _same_direction(right_most_wp, mwp_projected):
                        lane_poi.append(np.array([
                            -right_most_wp.transform.location.x,
                            right_most_wp.transform.location.y,
                            right_most_wp.transform.location.z
                        ]))
                        rloc_projected_prev = right_most_wp.previous(0.1)[0]
                        poi_prev = np.array([
                            -rloc_projected_prev.transform.location.x,
                            rloc_projected_prev.transform.location.y,
                            rloc_projected_prev.transform.location.z
                        ])
                        rloc_projected_next = right_most_wp.next(0.1)[0]
                        poi_next = np.array([
                            -rloc_projected_next.transform.location.x,
                            rloc_projected_next.transform.location.y,
                            rloc_projected_next.transform.location.z
                        ])
                        lane_prev_poi.append(poi_prev)
                        lane_next_poi.append(poi_next)
                        poi_wp_ids.add(str(right_most_wp.id))
                    else:
                        print(str(right_most_wp.id), "already in the set.")
                        break  # we hit a loop here
                elif lane_change_option == "Left":
                    break
            waypoints_np.append(wp1_inv)
        waypoints_np.append(wp2_inv)

        lane_poi = np.array(lane_poi)
        lane_prev_poi = np.array(lane_prev_poi)
        lane_next_poi = np.array(lane_next_poi)
        waypoints_np = np.array(waypoints_np)

        # 2. get the left and right drivable lanes according to the lane change option
        wps_dict = dict()
        for wp in waypoints:
            id = wp.id
            road_id = wp.road_id
            lane_id = wp.lane_id
            s = wp.s  # open drive parameters
            dict_id = "_".join([str(id), str(road_id), str(lane_id)])

            # set up for linked list traversal
            traversed_poi = set()
            lane_change_option = str(wp.lane_change)
            left_most_wp = wp
            temp_left_most_wp = left_most_wp
            while lane_change_option != 'NONE':
                if lane_change_option == "Both" or lane_change_option == "Left":
                    temp_left_most_wp = temp_left_most_wp.get_left_lane()
                    if not temp_left_most_wp:
                        print("There is no left lane the but the key of the previous lane is:", lane_change_option)
                        break  # there is no left lane but the key is not None
                    lane_change_option = str(temp_left_most_wp.lane_change)
                    if str(temp_left_most_wp.id) not in traversed_poi and _same_direction(temp_left_most_wp, wp):
                        left_most_wp = temp_left_most_wp
                        traversed_poi.add(str(temp_left_most_wp.id))
                    else:
                        print(str(temp_left_most_wp.id), "already in the set.")
                        break  # we hit a loop here
                elif lane_change_option == "Right":
                    break

            lane_change_option = str(wp.lane_change)
            right_most_wp = wp
            temp_right_most_wp = right_most_wp
            while lane_change_option != 'NONE':
                if lane_change_option == "Both" or lane_change_option == "Right":
                    temp_right_most_wp = temp_right_most_wp.get_right_lane()
                    if not temp_right_most_wp:
                        print("There is no right lane the but the key of the previous lane is:", lane_change_option)
                        break  # there is no right lane but the key is not None
                    lane_change_option = str(temp_right_most_wp.lane_change)
                    if str(temp_right_most_wp.id) not in traversed_poi and _same_direction(temp_right_most_wp, wp):
                        right_most_wp = temp_right_most_wp
                        traversed_poi.add(str(temp_right_most_wp.id))
                    else:
                        print(str(temp_right_most_wp.id), "already in the set.")
                        break  # we hit a loop here
                elif lane_change_option == "Left":
                    break
            wps_dict[dict_id] = {
                "s": s,
                "wp": wp,
                "left_most_wp": left_most_wp,
                "right_most_wp": right_most_wp
            }

        # 3. generate the rasterized surface and lane boundary coordinates
        lane_rasterized_coordinates = {"leftCoord": [], "rightCoord": [],
                                       "leftBoundCoord": [], "rightBoundCoord": []}
        for wp_id in wps_dict:
            left_side, right_side, \
                left_side_bound, right_side_bound = self.calculate_outreach_from_normal(wps_dict[wp_id]["wp"])

            # if we can change left, we expend our boundary
            if wps_dict[wp_id]["left_most_wp"].id != wps_dict[wp_id]["wp"].id:
                left_left_side, _, \
                    left_left_side_bound, _ = self.calculate_outreach_from_normal(wps_dict[wp_id]["left_most_wp"])
                left_side = left_left_side
                left_side_bound = left_left_side_bound

            # if we can change right, we expand our boundary
            if wps_dict[wp_id]["right_most_wp"].id != wps_dict[wp_id]["wp"].id:
                _, right_right_side, \
                    _, right_right_side_bound = self.calculate_outreach_from_normal(wps_dict[wp_id]["right_most_wp"])
                right_side = right_right_side
                right_side_bound = right_right_side_bound

            lane_rasterized_coordinates["leftCoord"].append(left_side)
            lane_rasterized_coordinates["rightCoord"].append(right_side)
            lane_rasterized_coordinates["leftBoundCoord"].append(left_side_bound)
            lane_rasterized_coordinates["rightBoundCoord"].append(right_side_bound)

        # convert to np array
        lane_rasterized_coordinates["leftCoord"] = np.array(lane_rasterized_coordinates["leftCoord"])
        lane_rasterized_coordinates["rightCoord"] = np.array(lane_rasterized_coordinates["rightCoord"])
        lane_rasterized_coordinates["leftBoundCoord"] = np.array(lane_rasterized_coordinates["leftBoundCoord"])
        lane_rasterized_coordinates["rightBoundCoord"] = np.array(lane_rasterized_coordinates["rightBoundCoord"])

        # 4. convert all relevant polygon to shapely polygons
        assert len(lane_rasterized_coordinates["leftCoord"]) == len(lane_rasterized_coordinates["rightCoord"])
        assert len(lane_rasterized_coordinates["leftCoord"]) == len(lane_rasterized_coordinates["leftBoundCoord"])
        assert len(lane_rasterized_coordinates["rightCoord"]) == len(lane_rasterized_coordinates["rightBoundCoord"])
        rasterized_surface = []
        polygon_left_lane_bound = []
        polygon_right_lane_bound = []
        for idx in range(len(lane_rasterized_coordinates["leftCoord"]) - 1):
            coord0 = lane_rasterized_coordinates["leftCoord"][idx]
            coord1 = lane_rasterized_coordinates["leftCoord"][idx + 1]
            coord2 = lane_rasterized_coordinates["rightCoord"][idx + 1]
            coord3 = lane_rasterized_coordinates["rightCoord"][idx]
            rasterGrid = geometry.Polygon([coord0, coord1, coord2, coord3])
            rasterGrid = rasterGrid.buffer(0.1)
            rasterized_surface.append(rasterGrid)

            # get left boundary as an rectangle obstacle
            coord0 = lane_rasterized_coordinates["leftBoundCoord"][idx]
            coord1 = lane_rasterized_coordinates["leftBoundCoord"][idx + 1]
            coord2 = lane_rasterized_coordinates["leftCoord"][idx + 1]
            coord3 = lane_rasterized_coordinates["leftCoord"][idx]
            rasterGrid = geometry.Polygon([coord0, coord1, coord2, coord3])
            rasterGrid = rasterGrid.buffer(0.1)
            polygon_left_lane_bound.append(rasterGrid)

            # get right boundary as an rectangle obstacle
            coord0 = lane_rasterized_coordinates["rightBoundCoord"][idx]
            coord1 = lane_rasterized_coordinates["rightBoundCoord"][idx + 1]
            coord2 = lane_rasterized_coordinates["rightCoord"][idx + 1]
            coord3 = lane_rasterized_coordinates["rightCoord"][idx]
            rasterGrid = geometry.Polygon([coord0, coord1, coord2, coord3])
            rasterGrid = rasterGrid.buffer(0.1)
            polygon_right_lane_bound.append(rasterGrid)

        if self.visualize:
            # plot rasterized lane surfaces and polygons
            for surface, leftBound, rightBound in zip(rasterized_surface, polygon_left_lane_bound,
                                                      polygon_right_lane_bound):
                xs, ys = surface.exterior.xy
                plt.fill(xs, ys, alpha=0.1, fc='blue', ec='None')
                xs, ys = leftBound.exterior.xy
                plt.fill(xs, ys, alpha=0.9, fc='green', ec='None')
                xs, ys = rightBound.exterior.xy
                plt.fill(xs, ys, alpha=0.9, fc='orange', ec='None')

            # plot waypoint line and POI
            plt.plot(waypoints_np[:, 0], waypoints_np[:, 1], color='blue')
            plt.scatter(lane_poi[1:-1, 0], lane_poi[1:-1, 1], marker='o', color='red')
            plt.scatter(lane_poi[0, 0], lane_poi[0, 1], marker='x', color='red')
            plt.scatter(lane_poi[-1, 0], lane_poi[-1, 1], marker='o', color='purple')
            plt.show()

        # 5. build final solution
        rasterized_optimized_lanes_around_traj["processedLane"] = {}
        rasterized_optimized_lanes_around_traj["processedLane"][0] = {
            "lanePOI": lane_poi,
            "lanePOINext": lane_next_poi,
            "lanePOIPrev": lane_prev_poi,
            "laneRasterizedCoords": lane_rasterized_coordinates,
            "rasterizedSurface": rasterized_surface,
            "polygonRightLaneBound": polygon_right_lane_bound,
            "polygonLeftLaneBound": polygon_left_lane_bound
        }
        # we only have one lane
        rasterized_optimized_lanes_around_traj["uniqueLaneIDs"] = [0]
        self.rasterized_optimized_lanes_around_traj = rasterized_optimized_lanes_around_traj

    def obstacles_on_lane(self, bbox_global_list):
        coord0 = bbox_global_list[0][0:2]
        coord1 = bbox_global_list[2][0:2]
        coord2 = bbox_global_list[6][0:2]
        coord3 = bbox_global_list[4][0:2]
        world_bbox_poly = geometry.Polygon([coord0, coord1, coord2, coord3])
        rasterized_surface = self.rasterized_optimized_lanes_around_traj["processedLane"][0]["rasterizedSurface"]
        for surface in rasterized_surface:
            if not surface.disjoint(world_bbox_poly):
                return True
        return False

    def collect_dynamic_obstacles(self, ego_vehicle, timestamp):
        dynamic_offLane_obstacles_list = list()
        dynamic_onLane_obstacles_list = list()
        ego_location = CarlaDataProvider.get_location(ego_vehicle)

        # get active actors, these will have IDs associated with them
        for vehicle in self.world.get_actors().filter('vehicle.*'):
            # if actor is alive and actor is not the ego vehicle
            if vehicle.is_alive and vehicle.get_location().distance(ego_location) > 0.1:
                active_obstacle_attr = dict()
                active_obstacle_attr["Id"] = vehicle.id
                active_obstacle_attr["dimension"] = np.array([
                    abs(vehicle.bounding_box.extent.x) * 2,
                    abs(vehicle.bounding_box.extent.y) * 2,
                    abs(vehicle.bounding_box.extent.z) * 2
                ])
                active_obstacle_attr["location"] = np.array([
                    -vehicle.get_transform().location.x,
                    vehicle.get_transform().location.y,
                    vehicle.get_transform().location.z
                ])
                active_obstacle_attr["rotation"] = np.array([
                    vehicle.get_transform().rotation.pitch,
                    vehicle.get_transform().rotation.yaw,
                    -vehicle.get_transform().rotation.roll
                ])
                bbox_local = vehicle.bounding_box.get_local_vertices()
                bbox_local_list = list()
                for vertex in bbox_local:
                    bbox_local_list.append(np.array([
                        -vertex.x,
                        vertex.y,
                        vertex.z
                    ]))
                active_obstacle_attr["bboxLocalVertices"] = bbox_local_list
                bbox_global = vehicle.bounding_box.get_world_vertices(vehicle.get_transform())
                bbox_global_list = list()
                for vertex in bbox_global:
                    bbox_global_list.append(np.array([
                        -vertex.x,
                        vertex.y,
                        vertex.z
                    ]))
                active_obstacle_attr["bboxWorldVertices"] = bbox_global_list
                if self.obstacles_on_lane(bbox_global_list):
                    dynamic_onLane_obstacles_list.append(active_obstacle_attr)
                else:
                    dynamic_offLane_obstacles_list.append(active_obstacle_attr)

        # get active actors, these will have IDs associated with them
        for walker in (CarlaDataProvider.get_world()).get_actors().filter('walker.*'):
            # if actor is alive
            if walker.is_alive:
                active_obstacle_attr = dict()
                active_obstacle_attr["id"] = walker.id
                active_obstacle_attr["dimension"] = np.array([
                    abs(walker.bounding_box.extent.x) * 2,
                    abs(walker.bounding_box.extent.y) * 2,
                    abs(walker.bounding_box.extent.z) * 2
                ])
                active_obstacle_attr["location"] = np.array([
                    -walker.get_transform().location.x,
                    walker.get_transform().location.y,
                    walker.get_transform().location.z
                ])
                active_obstacle_attr["rotation"] = np.array([
                    walker.get_transform().rotation.pitch,
                    walker.get_transform().rotation.yaw,
                    -walker.get_transform().rotation.roll
                ])
                active_obstacle_attr["transformMatrix"] = walker.get_transform().get_matrix()
                active_obstacle_attr["invTransformMatrix"] = walker.get_transform().get_inverse_matrix()
                bbox_local = walker.bounding_box.get_local_vertices()
                bbox_local_list = list()
                for vertex in bbox_local:
                    bbox_local_list.append(np.array([
                        -vertex.x,
                        vertex.y,
                        vertex.z
                    ]))
                active_obstacle_attr["bboxLocalVertices"] = bbox_local_list
                bbox_global = walker.bounding_box.get_world_vertices(walker.get_transform())
                bbox_global_list = list()
                for vertex in bbox_global:
                    bbox_global_list.append(np.array([
                        -vertex.x,
                        vertex.y,
                        vertex.z
                    ]))
                active_obstacle_attr["bboxWorldVertices"] = bbox_global_list
                if self.obstacles_on_lane(bbox_global_list):
                    dynamic_onLane_obstacles_list.append(active_obstacle_attr)
                else:
                    dynamic_offLane_obstacles_list.append(active_obstacle_attr)
        self.snapshot_obstacles["dynamicObstacles"][timestamp] = dict()
        self.snapshot_obstacles["dynamicObstacles"][timestamp]["offLaneObstacles"] = dynamic_offLane_obstacles_list
        self.snapshot_obstacles["dynamicObstacles"][timestamp]["onLaneObstacles"] = dynamic_onLane_obstacles_list

    def collect_static_obstacles(self):
        """
        This function needs to be called before the registration of the actors
        """
        # get static parked vehicle bounding box, we only consider parked vehicle, pedestrians are dynmaic
        static_vehicles = self.world.get_level_bbs(carla.CityObjectLabel.Vehicles)
        for bounding_box in static_vehicles:
            static_vehicle_attr = dict()
            static_vehicle_attr["Id"] = -1
            static_vehicle_attr["dimension"] = np.array([
                abs(bounding_box.extent.x) * 2,
                abs(bounding_box.extent.y) * 2,
                abs(bounding_box.extent.z) * 2
            ])
            static_vehicle_attr["location"] = np.array([
                -bounding_box.location.x,
                bounding_box.location.y,
                bounding_box.location.z
            ])
            static_vehicle_attr["rotation"] = np.array([
                bounding_box.rotation.pitch,
                bounding_box.rotation.yaw,
                -bounding_box.rotation.roll
            ])
            bbox_local = bounding_box.get_local_vertices()
            bbox_local_list = list()
            for vertex in bbox_local:
                bbox_local_list.append(np.array([
                    -vertex.x,
                    vertex.y,
                    vertex.z
                ]))
            static_vehicle_attr["bboxLocalVertices"] = bbox_local_list
            bbox_global = bounding_box.get_world_vertices(carla.Transform(carla.Location(), carla.Rotation()))
            bbox_global_list = list()
            for vertex in bbox_global:
                bbox_global_list.append(np.array([
                    -vertex.x,
                    vertex.y,
                    vertex.z
                ]))
            static_vehicle_attr["bboxWorldVertices"] = bbox_global_list
            self.snapshot_obstacles["staticObstacles"].append(static_vehicle_attr)

    def collect_ego_pose(self, ego_vehicle, timestamp):
        ego_vehicle_transform = ego_vehicle.get_transform()
        ego_telemetry_dict = dict()
        ego_telemetry_dict["location"] = np.array([
            -ego_vehicle_transform.location.x,
            ego_vehicle_transform.location.y,
            ego_vehicle_transform.location.z,
        ])
        ego_telemetry_dict["rotation"] = np.array([
            ego_vehicle_transform.rotation.pitch,
            ego_vehicle_transform.rotation.yaw,
            -ego_vehicle_transform.rotation.roll,
        ])
        # also find approximate way points
        ego_vehicle_location = ego_vehicle.get_transform().location
        ego_projected = self.map.get_waypoint(ego_vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        ego_next_wp = ego_projected.next(0.1)[0]
        ego_telemetry_dict["next"] = np.array([
            -ego_next_wp.transform.location.x,
            ego_next_wp.transform.location.y,
            ego_next_wp.transform.location.z,
        ])
        self.ego_telemetry[timestamp] = ego_telemetry_dict

    def save_collected_data(self):
        save_dict = {
            "rasterizedOptimizedLanesAroundTraj": self.rasterized_optimized_lanes_around_traj,
            "snapshotObstacles": self.snapshot_obstacles,
            "listOfTimestamp": self.timestamp,
            "egoTelemetry": self.ego_telemetry
        }
        if not os.path.exists(self.data_save_path):
            try:
                os.mkdir(self.data_save_path)
            except FileExistsError:
                pass
        with open(os.path.join(self.data_save_path, self.data_save_name), "wb") as f:
            pkl.dump(save_dict, f)
            print("Simulation saved.")

    def visualize_current_frame(self, visibleRange=70):
        # create figure
        fig, ax = plt.subplots(figsize=(3.5, 1.75))

        # get ego translation
        inverseEgoTranslate = -np.array(self.ego_telemetry[self.timestamp[-1]]["location"][0:2])
        inverseEgoRotate = -self.ego_telemetry[self.timestamp[-1]]["rotation"]
        inverseEgoAngle = inverseEgoRotate[1]
        inverseRotationMatrix = np.array([
            [np.cos(np.deg2rad(inverseEgoAngle)), -np.sin(np.deg2rad(inverseEgoAngle))],
            [np.sin(np.deg2rad(inverseEgoAngle)), np.cos(np.deg2rad(inverseEgoAngle))]
        ])

        # visualize lane surfaces boundaries and reachable POI
        rasterized_surface = self.rasterized_optimized_lanes_around_traj["processedLane"][0]["rasterizedSurface"]
        polygon_left_lane_bound = self.rasterized_optimized_lanes_around_traj["processedLane"][0]["polygonLeftLaneBound"]
        polygon_right_lane_bound = self.rasterized_optimized_lanes_around_traj["processedLane"][0]["polygonRightLaneBound"]
        for surface, leftBound, rightBound in zip(rasterized_surface, polygon_left_lane_bound, polygon_right_lane_bound):
            surface = translate(surface, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            surface = rotate(surface, angle=inverseEgoAngle, origin=(0, 0))
            xs, ys = surface.exterior.xy
            plt.fill(xs, ys, alpha=1.0, fc='#e0e0e0', ec='None')
            leftBound = translate(leftBound, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            leftBound = rotate(leftBound, angle=inverseEgoAngle, origin=(0, 0))
            xs, ys = leftBound.exterior.xy
            plt.fill(xs, ys, fc='#4d4d4d', ec='None')
            rightBound = translate(rightBound, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            rightBound = rotate(rightBound, angle=inverseEgoAngle, origin=(0, 0))
            xs, ys = rightBound.exterior.xy
            plt.fill(xs, ys, fc='#4d4d4d', ec='None')

        # plot waypoint line and POI
        lane_poi = copy.deepcopy(self.rasterized_optimized_lanes_around_traj["processedLane"][0]["lanePOI"])[:, 0:2]
        ego_location = self.ego_telemetry[self.timestamp[-1]]["location"]
        lane_poi[:, 0] = lane_poi[:, 0] + inverseEgoTranslate[0]
        lane_poi[:, 1] = lane_poi[:, 1] + inverseEgoTranslate[1]
        lane_poi = lane_poi @ inverseRotationMatrix.T
        plt.plot(lane_poi[:, 0], lane_poi[:, 1], '.r', markersize=2)

        # plot ego vehicle
        plt.plot(ego_location[0] + inverseEgoTranslate[0], ego_location[1] + inverseEgoTranslate[1], "vc", markersize=2)

        # visualize static obstacles
        for ob in self.snapshot_obstacles["staticObstacles"]:
            world_bbox_coord = ob["bboxWorldVertices"]
            coord0 = world_bbox_coord[0][0:2]
            coord1 = world_bbox_coord[2][0:2]
            coord2 = world_bbox_coord[6][0:2]
            coord3 = world_bbox_coord[4][0:2]
            world_bbox_poly = geometry.Polygon([coord0, coord1, coord2, coord3])
            world_bbox_poly = translate(world_bbox_poly, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            world_bbox_poly = rotate(world_bbox_poly, angle=inverseEgoAngle, origin=(0, 0))
            xs, ys = world_bbox_poly.exterior.xy
            plt.fill(xs, ys, alpha=0.9, fc='black', ec='black')

        # visualize dynamic obstacles
        for ob in self.snapshot_obstacles["dynamicObstacles"][self.timestamp[-1]]["offLaneObstacles"]:
            world_bbox_coord = ob["bboxWorldVertices"]
            coord0 = world_bbox_coord[0][0:2]
            coord1 = world_bbox_coord[2][0:2]
            coord2 = world_bbox_coord[6][0:2]
            coord3 = world_bbox_coord[4][0:2]
            world_bbox_poly = geometry.Polygon([coord0, coord1, coord2, coord3])
            world_bbox_poly = translate(world_bbox_poly, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            world_bbox_poly = rotate(world_bbox_poly, angle=inverseEgoAngle, origin=(0, 0))
            xs, ys = world_bbox_poly.exterior.xy
            plt.fill(xs, ys, alpha=1.0, fc='#ef8a62', ec='#ef8a62')

        for ob in self.snapshot_obstacles["dynamicObstacles"][self.timestamp[-1]]["onLaneObstacles"]:
            world_bbox_coord = ob["bboxWorldVertices"]
            coord0 = world_bbox_coord[0][0:2]
            coord1 = world_bbox_coord[2][0:2]
            coord2 = world_bbox_coord[6][0:2]
            coord3 = world_bbox_coord[4][0:2]
            world_bbox_poly = geometry.Polygon([coord0, coord1, coord2, coord3])
            world_bbox_poly = translate(world_bbox_poly, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            world_bbox_poly = rotate(world_bbox_poly, angle=inverseEgoAngle, origin=(0, 0))
            xs, ys = world_bbox_poly.exterior.xy
            plt.fill(xs, ys, alpha=1.0, fc='#b2182b', ec='#b2182b')

        plt.xlim(ego_location[0] + inverseEgoTranslate[0] - visibleRange, ego_location[0] + inverseEgoTranslate[0] + visibleRange)
        plt.ylim(ego_location[1] + inverseEgoTranslate[1] - visibleRange / 2, ego_location[1] + inverseEgoTranslate[1] + visibleRange / 2)
        ax.axis("off")
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        if not os.path.isdir(os.path.join("{}/bev_visualizations".format(BASE_PATH), self.data_save_name)):
            os.mkdir(os.path.join("{}/bev_visualizations".format(BASE_PATH), self.data_save_name))
        plt.savefig(os.path.join("{}/bev_visualizations".format(BASE_PATH), self.data_save_name,
                                 "{}.jpg".format(self.timestamp[-1])))
        plt.close()
