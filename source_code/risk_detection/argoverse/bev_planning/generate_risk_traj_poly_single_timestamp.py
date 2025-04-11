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
import os
import time
import path_configs

from concurrent.futures import ThreadPoolExecutor
import matplotlib as mpl

mpl.use('Agg')
print(mpl.get_backend())
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import random
import pickle as pkl
import logging
from datetime import datetime
from frenet_hyperparameters import STATIC_FOT_HYPERPARAMETERS
from argoverse.utils.interpolate import interp_arc
from shapely import geometry
from shapely.affinity import translate, rotate
from scipy.spatial.transform import Rotation as R
from planners import RRTStarPlanner, HybridAStarPlanner, FOTPlanner

from generate_bev import GenerateBEVGraph

logging.basicConfig(level=logging.ERROR)

# Area calculation
from math import radians
import shapely.geometry as sgeo

from kinematic_model import KinematicBicycleModel
from car_description import CarDescription

import numpy as np
import math
import os
import concurrent.futures
import time
import numpy as np
from scipy.spatial import Delaunay
from shapely.ops import cascaded_union, polygonize, unary_union


class Car:
    def __init__(self, init_x, init_y, init_v, init_yaw, delta_time):

        # Model parameters
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.velocity = init_v
        self.delta_time = delta_time
        self.time = 0.0
        self.wheel_angle = 0.0
        self.angular_velocity = 0.0
        self.acceleration = 0.0
        max_steer = radians(50)     # 33 degrees => radians
        wheelbase = 2.96

        # Car parameters
        overall_length = 4.97
        overall_width = 1.964
        tyre_diameter = 0.4826
        tyre_width = 0.265
        axle_track = 1.7
        rear_overhang = 0.5 * (overall_length - wheelbase)

        self.kinematic_bicycle_model = KinematicBicycleModel(wheelbase, max_steer, self.delta_time)
        self.description = CarDescription(overall_length, overall_width, rear_overhang, tyre_diameter, tyre_width, axle_track, wheelbase)

    def plot_car(self):
        return self.description.plot_car(self.x, self.y, self.yaw, self.wheel_angle)

    def drive(self, acceleration, steerAngle):
        self.acceleration = acceleration
        self.wheel_angle = steerAngle
        self.x, self.y, self.yaw, self.velocity, _, _ = self.kinematic_bicycle_model.update(self.x, self.y, self.yaw, self.velocity, self.acceleration, self.wheel_angle)

    def setStates(self, init_x, init_y, init_v, init_yaw):
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.velocity = init_v


def alpha_shape(points, alpha):
    """
    Compute the alpha shape of a set of points.

    Parameters:
    - points: list of (x,y) pairs representing the points.
    - alpha: alpha value.

    Returns:
    - Geometries representing the alpha shape.
    """
    if len(points) < 4:
        return sgeo.Polygon()

    # Delaunay triangulation
    tri = Delaunay(points)

    # Triangle vertices
    triangles = points[tri.vertices]

    # Alpha criterion: Circumradius must be smaller than alpha
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]

    # Lengths of triangle sides
    a_b = np.linalg.norm(a - b, axis=1)
    b_c = np.linalg.norm(b - c, axis=1)
    c_a = np.linalg.norm(c - a, axis=1)

    # Semiperimeter
    s = (a_b + b_c + c_a) / 2.0

    # Area
    area = np.sqrt(s * (s - a_b) * (s - b_c) * (s - c_a))

    # Circumradius
    circum_r = a_b * b_c * c_a / (4.0 * area)

    # Filter triangles
    valid = circum_r < alpha

    # Create the edges
    edges = []
    for i in range(triangles.shape[0]):
        if valid[i]:
            edge1 = [triangles[i, 0], triangles[i, 1]]
            edge2 = [triangles[i, 1], triangles[i, 2]]
            edge3 = [triangles[i, 2], triangles[i, 0]]
            edges.append(edge1)
            edges.append(edge2)
            edges.append(edge3)

    # Create a MultiLineString from the edges and extract polygons
    m = sgeo.MultiLineString(edges)
    triangles = list(polygonize(m))

    return cascaded_union(triangles)



def collided_single(egoCar, dynamic_obstacle, radius, radius_actor):
    if not dynamic_obstacle:
        return False

    buffered_origin = sgeo.Point(egoCar.x, egoCar.y).buffer(radius)
    return buffered_origin.intersects(dynamic_obstacle.buffer(radius_actor))


def collided(egoCar, dynamic_obstacles, curr_timestamp, radius, radius_actor):
    hit_cars_uuid = []
    for uuid, dynamic_obstacle in dynamic_obstacles[curr_timestamp].items():
        if collided_single(egoCar, dynamic_obstacle, radius, radius_actor):
            hit_cars_uuid.append(uuid)

    return len(hit_cars_uuid) != 0, hit_cars_uuid


class GenerateRiskArgoverse:
    def __init__(self, mapDataPath, trajDataPath, logID, plannerType, suffix,
                 routeVisDir, riskVisDir, riskSaveDir, concurrentObjCount,
                 plannerSeed=0, visualize=False, posUnc="None"):
        self.bevGraph = GenerateBEVGraph(mapDataPath=mapDataPath, trajDataPath=trajDataPath, logID=logID)
        self.rasterizedOptimizedLanesAroundTraj, \
        self.listofObstacles = self.bevGraph.buildOccupancyGrid(visualize=None)
        self.plannerType = plannerType
        self.routeVisDir = routeVisDir
        self.riskVisDir = riskVisDir
        self.riskSaveDir = riskSaveDir
        self.plannerSeed = plannerSeed
        self.visualize = visualize
        self.posUnc = posUnc
        self.posUncConfig = None
        self.concurrentObjCount = concurrentObjCount
        if self.posUnc != "None":
            print("Positional uncertainty enabled, reading configs from positional_uncertainty.json.")
            with open("./positional_uncertainty.json") as f:
                self.posUncConfig = json.load(f)
            assert self.posUnc in self.posUncConfig
            self.posUncConfig = self.posUncConfig[self.posUnc]
            print("Uncertainty parameters:", self.posUncConfig)
        else:
            print("Positional uncertainty is not enabled for this run.")
        if self.plannerSeed:
            np.random.seed(self.plannerSeed)
        self.suffix = suffix

    def riskAnalyserSingleTimestamp(self, timestamp, saveFileName=None, lookForwardDist=120, lookForwardTime=3):
        bookKeepingPerScene = dict()
        if timestamp in self.bevGraph.listofTimestamps:
            raArguement = dict()
            raArguement["timestamp"] = timestamp
            raArguement["lookForwardDist"] = lookForwardDist
            raArguement["lookForwardTime"] = lookForwardTime
            result = self._riskAnalyserOneTimestamp(raArguement)
            riskDict = result[0]
            riskFactorDict = result[1]
            riskFactorRawDict = result[2]
            dynamicObstaclesUUID = result[3]
            dynamicObstaclesPoly = result[4]
            bookKeepingPerScene[timestamp] = {
                "riskDict": riskDict,
                "riskFactorDict": riskFactorDict,
                "riskFactorRawDict": riskFactorRawDict,
                "dynamicObstaclesUUID": dynamicObstaclesUUID,
                "dynamicObstaclesPoly": dynamicObstaclesPoly
            }
        if self.riskSaveDir and saveFileName:
            # save to the tmp for intermediate results
            with open(os.path.join(self.riskSaveDir, "tmp_{}".format(self.suffix), saveFileName), "wb") as f:
                pkl.dump(bookKeepingPerScene, f)
                f.flush()
            print("Saved analysis pkl file to:",
                  os.path.join(self.riskSaveDir, "tmp_{}".format(self.suffix), saveFileName))

    def callPlanner(self):
        if self.plannerType == "rrt*":
            return RRTStarPlanner
        elif self.plannerType == "hybridA*":
            return HybridAStarPlanner
        elif self.plannerType == "fot*":
            return FOTPlanner
        else:
            raise NotImplementedError

    @staticmethod
    def _obstacleInLane(rasterizedLanes, polygontoTest):
        """
        A revised version of the _obstacleInLane used in generating BEV
        """
        for uniqueID in rasterizedLanes["uniqueLaneIDs"]:
            rasterizedSurfaces = rasterizedLanes["processedLane"][uniqueID]["rasterizedSurface"]
            for laneSurface in rasterizedSurfaces:
                if not laneSurface.disjoint(polygontoTest):
                    return True
        return False

    @staticmethod
    def _getAverageTimeTick(listOfTime):
        diff = list()
        for tdx in range(len(listOfTime) - 1):
            diff.append(listOfTime[tdx + 1] - listOfTime[tdx])
        return np.mean(diff) / 1e9

    @staticmethod
    def _getMedianTimeTick(listOfTime):
        diff = list()
        for tdx in range(len(listOfTime) - 1):
            diff.append(listOfTime[tdx + 1] - listOfTime[tdx])
        return np.median(diff) / 1e9

    def _getTerminalSpeed(self, listOfTime):
        trans1 = self.bevGraph.logEgoPoseDict[self.bevGraph.currentTrajID][listOfTime[-1]]["translation"]
        trans2 = self.bevGraph.logEgoPoseDict[self.bevGraph.currentTrajID][listOfTime[-2]]["translation"]
        terminalSpeed = np.linalg.norm(trans1 - trans2) / (listOfTime[-1] - listOfTime[-2]) * 1e9
        return terminalSpeed

    def _getInitialSpeed(self, listOfTime):
        trans1 = self.bevGraph.logEgoPoseDict[self.bevGraph.currentTrajID][listOfTime[1]]["translation"]
        trans2 = self.bevGraph.logEgoPoseDict[self.bevGraph.currentTrajID][listOfTime[0]]["translation"]
        initialSpeed = np.linalg.norm(trans1 - trans2) / (listOfTime[1] - listOfTime[0]) * 1e9
        return initialSpeed

    @staticmethod
    def _getLanePolygons(rasterizedLanes):
        """
        A revised version of the _obstacleInLane used in generating BEV
        """

        return [rasterizedLanes["processedLane"][uniqueID]["rasterizedSurface"] for uniqueID in
                rasterizedLanes["uniqueLaneIDs"]]


    @staticmethod
    def _getTimeBasedObstacles(listOfTime, staticObstaclesPoly,
                               dynamicObstaclesPoly, dynamicObstaclesUUID, remove=[]):
        timeBasedObstacles = dict()
        timeBasedObstacles["static"] = list()
        timeBasedObstacles["dynamic"] = list()
        if not dynamicObstaclesPoly or not dynamicObstaclesUUID:
            for _ in listOfTime:
                timeBasedObstacles["static"].append(staticObstaclesPoly)
                timeBasedObstacles["dynamic"].append({})
        else:
            for t in range(len(listOfTime)):
                timeBasedObstacles["static"].append(staticObstaclesPoly)
                assert len(dynamicObstaclesPoly[t]) == len(dynamicObstaclesUUID[t])
                currentTimeObs = dict()
                for odx, o in enumerate(dynamicObstaclesPoly[t]):
                    if dynamicObstaclesUUID[t][odx] not in remove:
                        currentTimeObs[dynamicObstaclesUUID[t][odx]] = o
                timeBasedObstacles["dynamic"].append(currentTimeObs)
        return timeBasedObstacles

    @staticmethod
    def _getTimeBasedObstaclesOrin(listOfTime, staticObstaclesPolyOrin,
                                   dynamicObstaclesPolyOrin, dynamicObstaclesUUID, remove=[]):
        timeBasedObstacles = dict()
        timeBasedObstacles["static"] = list()
        timeBasedObstacles["dynamic"] = list()
        if not dynamicObstaclesPolyOrin or not dynamicObstaclesUUID:
            for _ in listOfTime:
                timeBasedObstacles["static"].append(staticObstaclesPolyOrin)
                timeBasedObstacles["dynamic"].append({})
        else:
            for t in range(len(listOfTime)):
                timeBasedObstacles["static"].append(staticObstaclesPolyOrin)
                assert len(dynamicObstaclesPolyOrin[t]) == len(dynamicObstaclesUUID[t])
                currentTimeObs = dict()
                for odx, o in enumerate(dynamicObstaclesPolyOrin[t]):
                    if dynamicObstaclesUUID[t][odx] not in remove:
                        currentTimeObs[dynamicObstaclesUUID[t][odx]] = o
                timeBasedObstacles["dynamic"].append(currentTimeObs)
        return timeBasedObstacles

    def _convertPOItoRefPath(self, egoPosition, POI, travelDistFrac=0.1):
        _, _, egoCenterLine = self.bevGraph.argoverseMap.get_nearest_centerline(
            egoPosition, self.bevGraph.cityName, False)
        egoCenterLine = interp_arc(50, egoCenterLine[:, 0], egoCenterLine[:, 1])
        egoNearestPoint = egoCenterLine[0]
        egoNearestIdx = 0
        minDist = sys.float_info.max
        for idx, point in enumerate(egoCenterLine):
            dist = np.linalg.norm(point - egoPosition[0:2])
            if dist < minDist:
                egoNearestPoint = point
                egoNearestIdx = idx
                minDist = dist
        logging.debug("Nearest point to ego is: {}, at index {}.".format(egoNearestPoint, egoNearestIdx))
        if egoNearestIdx == len(egoCenterLine) - 1:
            egoNearestIdx -= 1
        egoLaneVector = egoCenterLine[egoNearestIdx + 1] - egoCenterLine[egoNearestIdx]
        unitEgoLaneVector = egoLaneVector / np.linalg.norm(egoLaneVector)
        egoToDest = POI[0:2] - egoPosition[0:2]
        distToDest = np.dot(unitEgoLaneVector, egoToDest)
        scaledEgoLaneVector = unitEgoLaneVector * distToDest * travelDistFrac

        _, _, poiCenterLine = self.bevGraph.argoverseMap.get_nearest_centerline(
            POI[0:2], self.bevGraph.cityName, False)
        poiCenterLine = interp_arc(50, poiCenterLine[:, 0], poiCenterLine[:, 1])
        poiNearestPoint = poiCenterLine[0]
        poiNearestIdx = 0
        minDist = sys.float_info.max
        for idx, point in enumerate(poiCenterLine):
            dist = np.linalg.norm(point - POI[0:2])
            if dist < minDist:
                poiNearestPoint = point
                poiNearestIdx = idx
                minDist = dist
        logging.debug("Nearest point to POI is: {}, at index {}.".format(poiNearestPoint, poiNearestIdx))
        if poiNearestIdx == 0:
            poiNearestIdx += 1
        poiLaneVector = poiCenterLine[poiNearestIdx - 1] - poiCenterLine[poiNearestIdx]
        unitPoiLaneVector = poiLaneVector / np.linalg.norm(poiLaneVector)
        destToEgo = egoPosition[0:2] - POI[0:2]
        distToEgo = np.dot(unitPoiLaneVector, destToEgo)
        scaledPoiLaneVector = unitPoiLaneVector * distToEgo * travelDistFrac

        # finally construct way points
        interPoint1 = egoPosition[0:2] + scaledEgoLaneVector
        interPoint2 = POI[0:2] + scaledPoiLaneVector

        logging.debug("The reference path is:", [egoPosition[0:2], interPoint1, interPoint2, POI[0:2]])
        return [egoPosition[0:2], interPoint1, interPoint2, POI[0:2]]

    def _find_reachable_area(self, funcArguments):
        start_preg = time.time()  # Debug: Calculating risk
        timeBasedObstacles = funcArguments["timeBasedObstacles"]
        timeBasedObstaclesOrin = funcArguments["timeBasedObstaclesOrin"]
        visualizeArea = funcArguments["visualizeArea"]
        origin = funcArguments["origin"]
        visualizeSaveName = funcArguments["visualizeSaveName"]
        removedList = funcArguments["removedList"]
        lanePolygonUnioned = funcArguments["lanePolygonUnioned"]
        initialSpeed = funcArguments["initialSpeed"]
        timestamp = funcArguments["timestamp"]
        return_visual_points = funcArguments.get("return_visual_points", False)

        # POC function parameters
        # obstacles, name, missing = None, empty_area = 0, areas = [], folder = "", save_plt = False, verbose = False
        dynamic_obstacles = timeBasedObstaclesOrin["dynamic"]
        dynamic_obstacles_buffered = timeBasedObstacles["dynamic"]

        name = visualizeSaveName
        missing = removedList
        # empty_area = 0, areas = [], folder = "",
        averageTimeTick = 0.1  # TODO: pass in from outside

        # Simulation specifications
        timeDelta = 0.1
        frames_ratio = round(timeDelta / averageTimeTick)
        totalTime = 1.5

        # Car specifications, can be put into the input
        maxAcc = +8
        minAcc = -7
        maxSteerRight = 33
        initX, initY, initV, initYaw = origin[0], origin[1], initialSpeed, 0
        egoCar = Car(initX, initY, initV, initYaw, timeDelta)
        xStep = yStep = vStep = 0.2
        yawStep = radians(5)
        # street_radius = 5.0

        # Setting radius of car
        radius = 0.8
        radius_actor = 0.3
        # radius_actor = 5  # this is for rear ending sceanrio

        all_dynamic_obstacles = []
        steppingSet = {}
        steppingSet[-1] = {(int(initX / xStep), int(initY / yStep), int(initV / vStep), int(initYaw / yawStep)): 1}

        point_in_init = True
        hit_cars = dict()

        for calc_time in range(int(np.ceil(totalTime / timeDelta))):
            curr_timestamp_index = calc_time * frames_ratio
            curr_timestamp = curr_timestamp_index
            all_dynamic_obstacles += (dynamic_obstacles[curr_timestamp].values())
            steppingHisto = {}
            if calc_time - 1 not in steppingSet or steppingSet[calc_time - 1] == {}:
                print(f"[{name}] Last time step did not produce any possible points!")
                break
            for initCondSet in steppingSet[calc_time - 1]:
                currInitXIdx, currInitYIdx, currInitVIdx, currInitYawIdx = initCondSet
                currInitX, currInitY, currInitV, currInitYaw = currInitXIdx * xStep, currInitYIdx * yStep, currInitVIdx * vStep, currInitYawIdx * yawStep
                for steer in [-maxSteerRight, 0, maxSteerRight]:
                    # for acc in [minAcc, 0, maxAcc]:
                    for acc in [maxAcc]:
                        egoCar.setStates(currInitX, currInitY, currInitV, currInitYaw)
                        egoCar.drive(acc, radians(steer))

                        if egoCar.velocity >= 0 and abs(egoCar.yaw) <= np.pi / 2:
                            if lanePolygonUnioned.contains(sgeo.Point(egoCar.x, egoCar.y).buffer(radius)):
                                point_in_init = False
                                collide_t, hit_cars_uuid = collided(egoCar, dynamic_obstacles, curr_timestamp, radius,
                                                                    radius_actor)
                                if not collide_t:
                                    bin = (int(round(egoCar.x / xStep)), int(round(egoCar.y / yStep)),
                                           int(round(egoCar.velocity / vStep)), int(round(egoCar.yaw / yawStep)))
                                    steppingHisto[bin] = 1
                                else:
                                    if curr_timestamp in hit_cars:
                                        hit_cars[curr_timestamp].union(hit_cars_uuid)
                                    else:
                                        hit_cars[curr_timestamp] = set(hit_cars_uuid)


                            elif point_in_init:
                                bin = (int(round(egoCar.x / xStep)), int(round(egoCar.y / yStep)),
                                       int(round(egoCar.velocity / vStep)), int(round(egoCar.yaw / yawStep)))
                                steppingHisto[bin] = 1

                steppingSet[calc_time] = steppingHisto

        all_points = set()
        for timeStampSet in steppingSet.values():
            for initCondSet in timeStampSet:
                all_points.add((initCondSet[0] * xStep, initCondSet[1] * yStep))
        hit_cars_poly = []
        for t in hit_cars.keys():
            for uuid in hit_cars[t]:
                hit_cars_poly.append(dynamic_obstacles[t][uuid])
        hit_cars_unioned = unary_union(hit_cars_poly)
        res = 0.0
        hull_pts = ([0], [0])

        # alpha = 1.2
        alpha = 1.5
        # alpha = 0.7

        all_points_np = np.array(list(all_points))

        try:
            hull = alpha_shape(all_points_np, alpha)
        except Exception as e:
            print(e, "Failed to draw hull, generated an empty polygon instead.")
            hull = sgeo.Polygon()

        adjusted = False

        # Polygon
        if isinstance(hull, sgeo.Polygon):
            try:
                if not hull.is_empty:
                    hull_pts = hull.exterior.coords.xy
            except Exception as e:
                print(e)
                print("Hull list error.")

            res = hull.area
            if not isinstance(hit_cars_unioned, list):
                hit_cars_unioned = [hit_cars_unioned]
            for hit_car_unioned in hit_cars_unioned:
                if hull.intersects(hit_car_unioned):
                    res -= hull.intersection(hit_car_unioned.buffer(radius_actor)).area
                    adjusted = True

        # Line
        elif isinstance(hull, sgeo.LineString):
            hull_pts = hull.convex_hull.xy

        # Multipolygon
        elif isinstance(hull, sgeo.multipolygon.MultiPolygon) or isinstance(hull, sgeo.collection.GeometryCollection):
            hull_pts = []
            res = hull.area
            for geom in hull.geoms:
                hull_pts.append((geom.exterior.xy[0], geom.exterior.xy[1]))
                if not isinstance(hit_cars_unioned, list):
                    hit_cars_unioned = [hit_cars_unioned]
                for hit_car_unioned in hit_cars_unioned:
                    if geom.intersects(hit_car_unioned):
                        res -= geom.intersection(hit_car_unioned.buffer(radius_actor)).area
                        adjusted = True

        else:
            raise Exception(f"Unimplemented hull type: {type(hull)}")



        if visualizeArea:
            self.areaVisualizer(all_points,
                                timeBasedObstaclesOrin["static"],
                                all_dynamic_obstacles,
                                hull_pts,
                                lanePolygonUnioned,
                                self.routeVisDir,
                                visualizeSaveName
                                )
        done_time = time.time()
        print("$$$$$$$$$$$$$$$$$$$$$ {}$$$$$$$$$$$$$$$$$$$$$$$$$$$$".format(done_time - start_preg))

        return max(res, 0.0), all_points_np, hull_pts

    # TODO we could get rid of the min distance as 5 because it cannot be reached
    # anyway if it is too close to the ego vehicle, and add a parameter T to specify
    # the look forward time
    def _riskAnalyserOneTimestamp(self, raArguement):
        random.seed(datetime.now())
        timestamp = raArguement["timestamp"]
        lookForwardDist = raArguement["lookForwardDist"]
        lookForwardTime = raArguement["lookForwardTime"]
        minDist = 5
        distanceToStaOb = 0.1
        distanceToDynOb = 1.5
        print("Start Processing Time stamp: {}".format(timestamp))
        startTime = time.time()

        # 0. get essential information for translation and rotation and a list of relavant timestamps
        poseCitytoEgo = self.bevGraph.logEgoPoseDict[self.bevGraph.currentTrajID][timestamp]
        inverseEgoTranslate = -np.array(poseCitytoEgo["translation"][0:2])
        inverseEgoRotate = poseCitytoEgo["rotation"].T
        r = R.from_matrix(inverseEgoRotate)
        inverseEgoAngle = r.as_euler('zxy', degrees=True)[0]
        listOfTime = [timestamp]
        for t in self.bevGraph.listofTimestamps:
            if t > timestamp:
                listOfTime.append(t)
                if (t - timestamp) / 1e9 >= lookForwardTime:
                    break
        if not len(listOfTime) or (listOfTime[-1] - timestamp) / 1e9 < lookForwardTime:
            if len(listOfTime):
                logging.warning("The maximum available look forward time is:", (listOfTime[-1] - timestamp) / 1e9)
            else:
                logging.warning("This is the last timestamp available in the log trajectory.")
            logging.warning("Return empty result.")
            return ({}, {}, {}, [], [])

        # taka list of dynamic obstacles that aligns with the time stamps
        dynamicObstacles = list()
        for t in listOfTime:
            dynamicObstacles.append(self.listofObstacles["dynamicObstacles"][t])



        # 2. origin as current position, list of static boundary obstacles, and goal as POI
        origin = np.array([0, 0])
        staticObstaclesPoly = []
        staticObstaclesPolyOrin = []
        leftLaneBoundaries = self.listofObstacles["laneBoundaries"]["leftLaneBoundaries"]
        for boundary in leftLaneBoundaries:
            boundary = translate(boundary, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            boundary = rotate(boundary, angle=inverseEgoAngle, origin=(0, 0))
            staticObstaclesPolyOrin.append(boundary)
            boundary = boundary.buffer(distanceToStaOb, cap_style=3)
            staticObstaclesPoly.append(boundary)
        rightLaneBoundaries = self.listofObstacles["laneBoundaries"]["rightLaneBoundaries"]
        for boundary in rightLaneBoundaries:
            boundary = translate(boundary, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            boundary = rotate(boundary, angle=inverseEgoAngle, origin=(0, 0))
            staticObstaclesPolyOrin.append(boundary)
            boundary = boundary.buffer(distanceToStaOb, cap_style=3)
            staticObstaclesPoly.append(boundary)
        frontLaneBoundaries = self.listofObstacles["laneBoundaries"]["frontLaneBoundaries"]
        for boundary in frontLaneBoundaries:
            boundary = translate(boundary, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            boundary = rotate(boundary, angle=inverseEgoAngle, origin=(0, 0))
            staticObstaclesPolyOrin.append(boundary)
            boundary = boundary.buffer(distanceToStaOb, cap_style=3)
            staticObstaclesPoly.append(boundary)
        rearLaneBoundaries = self.listofObstacles["laneBoundaries"]["rearLaneBoundaries"]
        for boundary in rearLaneBoundaries:
            boundary = translate(boundary, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            boundary = rotate(boundary, angle=inverseEgoAngle, origin=(0, 0))
            staticObstaclesPolyOrin.append(boundary)
            boundary = boundary.buffer(distanceToStaOb, cap_style=3)
            staticObstaclesPoly.append(boundary)

        # 3. delete a selected dynamic obstacle (starting with the ones on-lane)
        # instead of getting rid of an obstacle per time step, we need to get rid of an
        # obstacle and its entire trajectory up to time T, which means for t = 0 to t = T,
        # suppose the obstacle appears in a subset of the frames in this period, delete a selected 
        # obstacle means deleting it from all frames that contains it from 0 to T

        # all of the following becomes multiple timestemp
        # dynamicObstaclesXY = list()
        dynamicObstaclesPoly = list()
        dynamicObstaclesPolyOrin = list()
        dynamicObstaclesUUID = list()

        # the risk is still single timestamp
        riskDict = dict()
        riskFactorRawDict = dict()
        riskFactorDict = dict()

        # TODO the noise are applied for each time step from 0 to T as well
        for tdx in range(len(dynamicObstacles)):
            # currentDynamicObstaclesXY = list()
            currentDynamicObstaclesPoly = list()
            currentDynamicObstaclesPolyOrin = list()
            currentDynamicObstaclesUUID = dict()
            if self.posUnc == "None":
                logging.info("Positional uncertainty disabled in this frame.")
                allDynamicObstacles = dynamicObstacles[tdx]["onLaneObstacles"]
                for obstacle in allDynamicObstacles:
                    bboxWorldCoord = obstacle.bbox_city_fr
                    bboxWorldCoord = bboxWorldCoord[:, 0:2]
                    bboxPolygon = geometry.Polygon([bboxWorldCoord[0],
                                                    bboxWorldCoord[1],
                                                    bboxWorldCoord[3],
                                                    bboxWorldCoord[2]])
                    # we do not need to buffer it anymore, the frenet will take care of the buffering
                    bboxPolygon = translate(bboxPolygon, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
                    bboxPolygon = rotate(bboxPolygon, angle=inverseEgoAngle, origin=(0, 0))
                    currentDynamicObstaclesPolyOrin.append(bboxPolygon)
                    bboxPolygon = bboxPolygon.buffer(distanceToDynOb, cap_style=3)
                    currentDynamicObstaclesPoly.append(bboxPolygon)
                    currentDynamicObstaclesUUID[len(currentDynamicObstaclesPoly) - 1] = obstacle.track_uuid
            else:
                logging.error("Uncertainty type not supported.")
                raise NotImplementedError
            # dynamicObstaclesXY.append(currentDynamicObstaclesXY)
            dynamicObstaclesPoly.append(currentDynamicObstaclesPoly)
            dynamicObstaclesUUID.append(currentDynamicObstaclesUUID)
            dynamicObstaclesPolyOrin.append(currentDynamicObstaclesPolyOrin)

        # also transform lane surfaces
        for laneId in self.rasterizedOptimizedLanesAroundTraj['uniqueLaneIDs']:
            self.rasterizedOptimizedLanesAroundTraj['processedLane'][laneId]['rasterizedSurfaceTrans'] = list()
            for laneSurface in self.rasterizedOptimizedLanesAroundTraj['processedLane'][laneId]['rasterizedSurface']:
                laneSurfaceCopy = copy.deepcopy(laneSurface)
                laneSurfaceCopy = translate(laneSurfaceCopy, xoff=inverseEgoTranslate[0],
                                            yoff=inverseEgoTranslate[1])
                laneSurfaceCopy = rotate(laneSurfaceCopy, angle=inverseEgoAngle, origin=(0, 0))
                self.rasterizedOptimizedLanesAroundTraj['processedLane'][laneId]['rasterizedSurfaceTrans'].append(
                    laneSurfaceCopy)

        # Merge all rasterized surfaces into one
        lanePolygons = []
        for poly_lane in self._getLanePolygons(self.rasterizedOptimizedLanesAroundTraj):
            for poly in poly_lane:
                lanePolygons.append(poly)

        lanePolygonUnioned = cascaded_union(lanePolygons)
        lanePolygonUnioned = translate(lanePolygonUnioned, xoff=inverseEgoTranslate[0],
                                       yoff=inverseEgoTranslate[1])
        lanePolygonUnioned = rotate(lanePolygonUnioned, angle=inverseEgoAngle, origin=(0, 0))


        # here is a good place to generate hyperparameters for the frenet planner
        """
        1. hyperparameters that are frame and parameter independent: max speed, max acc, max cur, max road width,
           road width sample delta, targeted speed sample delta, sample number of speed, distance buffer to obstacle,
           threshold distance to goal, and all lost cost parameters.
        2. hyperparameters that are frame dependent:  time tick, and targeted speed (the end speed)
        """

        averageTimeTick = self._getAverageTimeTick(listOfTime)
        medianTimeTick = self._getMedianTimeTick(listOfTime)
        # we should not use terminal speed here because speed keeping is not the only mode
        # terminalSpeed = self._getTerminalSpeed(listOfTime)
        initialSpeed = self._getInitialSpeed(listOfTime)
        logging.info("Median time tick {}, average time tick {}.".format(medianTimeTick, averageTimeTick))
        logging.info("InitialSpeed {} m/s.".format(initialSpeed))


        fotInitialState = {
            "c_speed": initialSpeed,
            "c_d": 0,
            "c_d_d": 0,
            "c_d_dd": 0,
            "s0": 0
        }


        # 4. check if the goal is reachable or not with removing one obstacle per time
        emptySuccess = 0
        successGoals = list()
        timeBasedObstacles = self._getTimeBasedObstacles(listOfTime, staticObstaclesPoly, None, None)
        timeBasedObstaclesOrin = self._getTimeBasedObstaclesOrin(listOfTime, staticObstaclesPolyOrin, None, None)

        # prepare arguments for parallel execution

        emptyArgument = dict()
        emptyArgument["fotInitialState"] = fotInitialState
        emptyArgument["timeBasedObstacles"] = timeBasedObstacles
        emptyArgument["timeBasedObstaclesOrin"] = timeBasedObstaclesOrin
        emptyArgument["visualizeArea"] = False
        emptyArgument["origin"] = origin
        emptyArgument["visualizeSaveName"] = "{}_golden".format(timestamp)
        emptyArgument["removedList"] = []
        emptyArgument["lanePolygonUnioned"] = lanePolygonUnioned
        emptyArgument["initialSpeed"] = self._getInitialSpeed(listOfTime)
        emptyArgument["timestamp"] = timestamp
        reachable_area_empty, _, _ = self._find_reachable_area(emptyArgument)
        print(reachable_area_empty)
        if not emptySuccess:
            logging.warning("Warning there is no success but we will continue anyway.")
        print("Done evaluating empty grid, empty success:", emptySuccess, "succeed goals:", successGoals)


        timeBasedObstacles = self._getTimeBasedObstacles(listOfTime, staticObstaclesPoly,
                                                         dynamicObstaclesPoly, dynamicObstaclesUUID)
        timeBasedObstaclesOrin = self._getTimeBasedObstaclesOrin(listOfTime, staticObstaclesPolyOrin,
                                                                 dynamicObstaclesPolyOrin, dynamicObstaclesUUID)
        # prepare arguments for parallel execution
        fullArgument = dict()
        fullArgument["fotInitialState"] = fotInitialState
        fullArgument["timeBasedObstacles"] = timeBasedObstacles
        fullArgument["timeBasedObstaclesOrin"] = timeBasedObstaclesOrin
        fullArgument["visualizeArea"] = False
        fullArgument["origin"] = origin
        fullArgument["visualizeSaveName"] = "{}_full".format(timestamp)
        fullArgument["removedList"] = []
        fullArgument["lanePolygonUnioned"] = lanePolygonUnioned
        fullArgument["initialSpeed"] = self._getInitialSpeed(listOfTime)
        fullArgument["timestamp"] = timestamp
        fullArgument["return_visual_points"] = True


        reachable_area_full, all_points_np, hull_pts = self._find_reachable_area(fullArgument)
        print("Done evaluating fully occupied grid, full obstacle success:", reachable_area_full, "succeed goals:",
              successGoals)

        # 5. Finally remove obstacles one by one
        uuidCurrentTime = list(dynamicObstaclesUUID[0].values())
        for k in range(len(uuidCurrentTime)):
            dynamicObstaclesLessOnePoly = copy.deepcopy(dynamicObstaclesPoly)
            dynamicObstaclesLessOnePolyOrin = copy.deepcopy(dynamicObstaclesPolyOrin)
            uuid = uuidCurrentTime[k]
            print("UUID to be removed at time {} is {}.".format(timestamp, uuid))
            lessOneSuccess = 0
            timeBasedObstaclesLessOne = self._getTimeBasedObstacles(listOfTime,
                                                                    staticObstaclesPoly,
                                                                    dynamicObstaclesLessOnePoly,
                                                                    dynamicObstaclesUUID,
                                                                    [uuid])
            timeBasedObstaclesLessOneOrin = self._getTimeBasedObstaclesOrin(listOfTime,
                                                                            staticObstaclesPolyOrin,
                                                                            dynamicObstaclesLessOnePolyOrin,
                                                                            dynamicObstaclesUUID,
                                                                            [uuid])

            # prepare arguments for parallel execution

            lessOneArgument = dict()
            lessOneArgument["fotInitialState"] = fotInitialState
            lessOneArgument["timeBasedObstacles"] = timeBasedObstaclesLessOne
            lessOneArgument["timeBasedObstaclesOrin"] = timeBasedObstaclesLessOneOrin
            lessOneArgument["timeBasedObstaclesVis"] = timeBasedObstaclesOrin
            lessOneArgument["visualizeArea"] = False
            lessOneArgument["origin"] = origin
            lessOneArgument["visualizeSaveName"] = "{}_lessone_{}".format(timestamp, k)
            lessOneArgument["removedList"] = [uuid]
            lessOneArgument["lanePolygonUnioned"] = lanePolygonUnioned
            lessOneArgument["initialSpeed"] = self._getInitialSpeed(listOfTime)
            lessOneArgument["timestamp"] = timestamp

            reachable_area_lessone, _, _ = self._find_reachable_area(lessOneArgument)

            if reachable_area_empty:
                riskFactor = (reachable_area_lessone - reachable_area_full) * 1.0 / reachable_area_empty
            else:
                riskFactor = 0.0
            riskDict[str(k)] = reachable_area_lessone
            riskFactorRawDict[str(k)] = riskFactor
            riskFactor = max(riskFactor, 0)
            riskFactorDict[str(k)] = riskFactor
            del dynamicObstaclesLessOnePoly

        riskDict["empty"] = reachable_area_empty
        riskDict["full"] = reachable_area_full
        print("Done analysing one frame after {} seconds.".format(time.time() - startTime))
        print(riskFactorRawDict, riskFactorDict, riskDict, dynamicObstaclesUUID[0])

        if self.visualize:
            timeBasedObstaclesOrin = self._getTimeBasedObstaclesOrin(listOfTime, staticObstaclesPolyOrin,
                                                                     dynamicObstaclesPolyOrin, dynamicObstaclesUUID)

            if not hull_pts:
                print("Hull points is empty!")
            else:
                self.riskVisualizer(all_points_np, hull_pts, timeBasedObstaclesOrin["static"],
                                    timeBasedObstaclesOrin["dynamic"], riskFactorDict, dynamicObstaclesUUID,
                                    origin, self.riskVisDir, timestamp, lanePolygonUnioned)
        return riskDict, riskFactorDict, riskFactorRawDict, dynamicObstaclesUUID, dynamicObstaclesPoly

    @staticmethod
    def riskVisualizer(all_points_np, hull_pts, static_obstacles, dynamic_obstacles, riskFactorDict,
                       dynamicObstaclesUUID, location, savedir, timestamp, lanePolygonUnioned,
                       vis_left=-25, vis_right=25, vis_top=25, vis_bottom=-25):

        cmap = plt.cm.RdYlGn_r
        plt.figure(figsize=(15, 15), dpi=200)

        circle_buffered = sgeo.Point(location).buffer(0.8)
        x, y = circle_buffered.exterior.xy
        plt.fill(x, y, color="yellow")

        if isinstance(lanePolygonUnioned, sgeo.Polygon):
            try:
                if not lanePolygonUnioned.is_empty:
                    x, y = lanePolygonUnioned.exterior.coords.xy
                    plt.fill(x, y, zorder=-2, color="black", alpha=0.15)
            except Exception as e:
                print(e)
                print("visualize error")

        # Line
        elif isinstance(lanePolygonUnioned, sgeo.LineString):
            x, y = lanePolygonUnioned.convex_hull.xy
            plt.fill(x, y, zorder=-2, color="black", alpha=0.15)

        # Multipolygon
        elif isinstance(lanePolygonUnioned, sgeo.multipolygon.MultiPolygon) or isinstance(lanePolygonUnioned, sgeo.collection.GeometryCollection):
            res = lanePolygonUnioned.area
            for geom in lanePolygonUnioned.geoms:
                x, y = geom.exterior.xy
                plt.fill(x, y, zorder=-2, color="black", alpha=0.15)

            # hull_pts = np.array(hull_pts)





        for poly_list in static_obstacles:
            for poly in poly_list:
                x, y = poly.exterior.xy
                plt.fill(x, y, color="black")

        for idx, obstacles_t in enumerate(dynamic_obstacles):
            if idx == 0:
                # for uuid, obstacle in obstacles_t.items():
                #     dynamicObstaclesUUID[idx]
                #     riskColor = cmap(riskFactorDict[str(idx)])
                #     xs, ys = obstacle.exterior.xy
                #     plt.fill(xs, ys, alpha=1.0, fc=riskColor, ec=riskColor)
                #     plt.text(obstacle.centroid.x, obstacle.centroid.y, f'{round(riskFactorDict[str(idx)], 2)}', bbox=dict(facecolor='yellow', alpha=0.5), zorder=2)
                max_risk = sys.float_info.min
                for riskID in riskFactorDict:
                    obsRisk = riskFactorDict[riskID]
                    max_risk = max(max_risk, obsRisk)
                for riskID in riskFactorDict:
                    obsRisk = riskFactorDict[riskID]
                    obsUUID = dynamicObstaclesUUID[idx][int(riskID)]
                    obsPoly = obstacles_t[obsUUID]
                    if max_risk != 0:
                        riskColor = cmap(obsRisk / max_risk)
                    else:
                        riskColor = cmap(0)
                    xs, ys = obsPoly.exterior.xy
                    plt.fill(xs, ys, alpha=1.0, fc=riskColor, ec=riskColor)
                    if vis_left < obsPoly.centroid.x < vis_right and vis_bottom < obsPoly.centroid.y < vis_top:
                        # plt.text(obsPoly.centroid.x, obsPoly.centroid.y, f'{round(obsRisk, 2)}',
                        #         bbox=dict(facecolor='yellow', alpha=0.5), zorder=2)
                        pass
            elif obstacles_t != dict():
                for uuid, obstacle in obstacles_t.items():
                    xs, ys = obstacle.exterior.xy
                    plt.fill(xs, ys, alpha=0.2, fc="#999999", ec="#999999", zorder=-2)

        for point in all_points_np:
            plt.scatter(point[0], point[1], color='blue', s=0.5)

        if isinstance(hull_pts, tuple):
            plt.scatter(hull_pts[0], hull_pts[1], color='darkgreen')
            plt.fill(hull_pts[0], hull_pts[1], alpha=0.5, zorder=-2, color="green")
        elif isinstance(hull_pts, list):
            for hull_pts_tuple in hull_pts:
                plt.scatter(hull_pts_tuple[0], hull_pts_tuple[1], color='darkgreen')
                plt.fill(hull_pts_tuple[0], hull_pts_tuple[1], alpha=0.5, zorder=-2, color="green")
        else:
            raise Exception("Printing hull error. Timestamp: {}, index: {}".format(timestamp, timestampIdx))

        plt.xlim(vis_left, vis_right)
        plt.ylim(vis_bottom, vis_top)

        plt.savefig("{}/{}.jpg".format(savedir, timestamp), extent=[vis_left, vis_right, vis_bottom, vis_top], bbox_inches='tight')
        plt.close()

    def areaVisualizer(self, all_points_np, static_obstacles, all_dynamic_obstacles, hull_pts, lanePolygonUnioned,
                       savedir, savename, vis_left=-25, vis_right=25, vis_top=25, vis_bottom=-25):

        plt.figure(figsize=(15, 15), dpi=200)

        for poly_list in static_obstacles:
            for poly in poly_list:
                x, y = poly.exterior.xy
                plt.fill(x, y, color="black")

        circle_buffered = sgeo.Point(0, 0).buffer(0.8)
        x, y = circle_buffered.exterior.xy
        plt.fill(x, y, color="yellow")

        for point in all_points_np:
            plt.scatter(point[0], point[1], color='blue', s=0.5)

        for dynamic_bbox in all_dynamic_obstacles:
            x, y = dynamic_bbox.exterior.xy
            plt.fill(x, y, zorder=1, color="red", alpha=0.2)

        if isinstance(hull_pts, tuple):
            plt.scatter(hull_pts[0], hull_pts[1], color='darkgreen')
            plt.fill(hull_pts[0], hull_pts[1], alpha=0.5, zorder=1, color="green")
        elif isinstance(hull_pts, list):
            for hull_pts_tuple in hull_pts:
                plt.scatter(hull_pts_tuple[0], hull_pts_tuple[1], color='darkgreen')
                plt.fill(hull_pts_tuple[0], hull_pts_tuple[1], alpha=0.5, zorder=1, color="green")
        else:
            raise Exception("Printing hull error. Timestamp: {}, index: {}".format(timestamp, timestampIdx))

        plt.xlim(vis_left, vis_right)
        plt.ylim(vis_bottom, vis_top)

        x, y = lanePolygonUnioned.exterior.xy
        plt.fill(x, y, zorder=-2, color="green", alpha=0.3)

        plt.savefig("{}/{}.jpg".format(savedir, savename), bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    basePath = path_configs.BASEPATH
    mapDataPath = os.path.join(basePath, "map_files")
    subFolder = sys.argv[1]
    logID = sys.argv[2]
    plannerType = sys.argv[3]
    assert plannerType == "dyn", "Planner type for trajectory based risk analysis can only be dyn"
    suffix = sys.argv[4]
    posUnc = sys.argv[5]
    timestamp = int(sys.argv[6])
    concurrentObjCount = int(sys.argv[7])
    if posUnc == "None":
        prefix = "model_unc"
    elif posUnc in ["gaussian2DShift", "gaussian2DRotate", "gaussian2DCorners", "gaussian2DShiftRotate"]:
        if posUnc == "gaussian2DShift":
            prefix = "pos_unc_shift"
        if posUnc == "gaussian2DRotate":
            prefix = "pos_unc_rotate"
        if posUnc == "gaussian2DCorners":
            prefix = "pos_unc_cor"
        if posUnc == "gaussian2DShiftRotate":
            prefix = "pos_unc_shift_rotate"
    else:
        raise Exception("Not supported posUnc")
    trajDataPath = os.path.join(basePath, "argoverse-tracking", subFolder)
    routeVisDir = os.path.join(basePath, "visualize_routing_{}".format(plannerType), subFolder, logID, prefix, suffix)
    riskVisDir = os.path.join(basePath, "visualize_risk_{}".format(plannerType), subFolder, logID, prefix, suffix)
    riskSaveDir = os.path.join(basePath, "analysis_risk_{}".format(plannerType), subFolder, logID, prefix)
    if len(sys.argv) == 8:
        genRisk = GenerateRiskArgoverse(mapDataPath=mapDataPath,
                                        trajDataPath=trajDataPath,
                                        logID=logID,
                                        plannerType=plannerType,
                                        routeVisDir=routeVisDir,
                                        riskVisDir=riskVisDir,
                                        riskSaveDir=riskSaveDir,
                                        visualize=True,
                                        posUnc=posUnc,
                                        concurrentObjCount=concurrentObjCount,
                                        suffix=suffix)
    elif len(sys.argv) > 8:
        seed = sys.argv[8]
        genRisk = GenerateRiskArgoverse(mapDataPath=mapDataPath,
                                        trajDataPath=trajDataPath,
                                        logID=logID,
                                        plannerType=plannerType,
                                        routeVisDir=routeVisDir,
                                        riskVisDir=riskVisDir,
                                        riskSaveDir=riskSaveDir,
                                        plannerSeed=int(seed),
                                        visualize=True,
                                        posUnc=posUnc,
                                        concurrentObjCount=concurrentObjCount,
                                        suffix=suffix)
    else:
        raise ValueError
    genRisk.riskAnalyserSingleTimestamp(saveFileName="{}_risk_analysis.pkl.{}".format(suffix, timestamp),
                                        timestamp=timestamp)
