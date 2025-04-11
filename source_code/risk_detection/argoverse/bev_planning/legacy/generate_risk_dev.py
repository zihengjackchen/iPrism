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

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np
import json
import gc
import sys
import random
import pickle as pkl

from shapely.geometry.base import JOIN_STYLE
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.frame_label_accumulator import PerFrameLabelAccumulator
from argoverse.utils.geometry import rotate_polygon_about_pt, translate_polygon
from argoverse.utils.se3 import SE3
from argoverse.utils.mpl_plotting_utils import draw_lane_polygons, plot_bbox_2D
from argoverse.utils.interpolate import interp_arc
from shapely import geometry
from shapely.affinity import translate, rotate
from scipy.spatial.transform import Rotation as R
from planners import RRTStarPlanner, HybridAStarPlanner, FOTPlanner

from generate_bev import GenerateBEVGraph


# Generate risk analysis for one Ego trajectory (one logID)
class GenerateRisk:
    def __init__(self, mapDataPath, trajDataPath, logID, plannerType,
                 routeVisDir, riskVisDir, riskSaveDir, plannerSeed=0, visualize=False, posUnc="None"):
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
        if self.posUnc != "None":
            print("Positional uncertainty enabled, reading configs from positional_uncertainty.json.")
            with open("../positional_uncertainty.json") as f:
                self.posUncConfig = json.load(f)
            assert self.posUnc in self.posUncConfig
            self.posUncConfig = self.posUncConfig[self.posUnc]
            print("Uncertainty parameters:", self.posUncConfig)
        else:
            print("Positional uncertainty is not enabled for this run.")
        if self.plannerSeed:
            np.random.seed(self.plannerSeed)

    def riskAnalyser(self, saveFileName=None, lookForwardDist=50):
        if self.riskSaveDir and saveFileName:
            if os.path.isfile(os.path.join(self.riskSaveDir, saveFileName)):
                print("{} exists, already analysed, return...".format(os.path.join(self.riskSaveDir, saveFileName)))
                return
        bookKeepingPerScene = dict()
        for timestamp in self.bevGraph.listofTimestamps:
            riskDict, riskFactorDict, riskFactorRawDict, \
                dynamicObstaclesUUID, dynamicObstaclesXY = self._riskAnalyserOneTimestamp(timestamp, lookForwardDist)
            bookKeepingPerScene[timestamp] = {
                "riskDict": riskDict,
                "riskFactorDict": riskFactorDict,
                "riskFactorRawDict": riskFactorRawDict,
                "dynamicObstaclesUUID": dynamicObstaclesUUID,
                "dynamicObstaclesXY": dynamicObstaclesXY
            }
        if self.riskSaveDir and saveFileName:
            with open(os.path.join(self.riskSaveDir, saveFileName), "wb") as f:
                pkl.dump(bookKeepingPerScene, f)
            print("Saved analysis pkl file to:", os.path.join(self.riskSaveDir, saveFileName))

    def callPlanner(self):
        if self.plannerType == "rrt*":
            return RRTStarPlanner
        elif self.plannerType == "hybridA*":
            return HybridAStarPlanner
        elif self.plannerType == "fot":
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

    def _riskAnalyserOneTimestamp(self, timestamp, lookForwardDist, minDist=5):
        print("Time stamp: {}".format(timestamp))
        # 0. get essential information for translation and rotation
        poseCitytoEgo = self.bevGraph.logEgoPoseDict[self.bevGraph.currentTrajID][timestamp]
        inverseEgoTranslate = -np.array(poseCitytoEgo["translation"][0:2])
        inverseEgoRotate = poseCitytoEgo["rotation"].T
        r = R.from_matrix(inverseEgoRotate)
        inverseEgoAngle = r.as_euler('zxy', degrees=True)[0]
        dynamicObstacles = self.listofObstacles["dynamicObstacles"][timestamp]

        # 1. find the next set of POI based on the current location
        setOfGoals = []
        for laneID in self.rasterizedOptimizedLanesAroundTraj["uniqueLaneIDs"]:
            lanePOI = copy.deepcopy(self.rasterizedOptimizedLanesAroundTraj["processedLane"][laneID]["lanePOI"])
            if len(lanePOI):
                lanePOI[:, 0] = lanePOI[:, 0] + inverseEgoTranslate[0]
                lanePOI[:, 1] = lanePOI[:, 1] + inverseEgoTranslate[1]
                lanePOI = lanePOI @ inverseEgoRotate[:2, :2].T
                for POI in lanePOI:
                    if minDist < POI[0] <= lookForwardDist:
                        setOfGoals.append(list(POI))

        # 2. origin as current position, list of static boundary obstacles, and goal as POI
        setOfGoals = sorted(setOfGoals, key=lambda element: (element[0], element[1]))
        if len(setOfGoals) > 20:
            print("Restricting the number of goals to 20, currently the length is {}.".format(len(setOfGoals)))
            setOfGoals = setOfGoals[:20]
        setOfGoals = np.array(setOfGoals)
        origin = np.array([0, 0])
        staticObstacles = []
        leftLaneBoundaries = self.listofObstacles["laneBoundaries"]["leftLaneBoundaries"]
        for boundary in leftLaneBoundaries:
            boundary = translate(boundary, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            boundary = rotate(boundary, angle=inverseEgoAngle, origin=(0, 0))
            xs, ys = boundary.exterior.xy
            xmax, xmin = np.max(xs), np.min(xs)
            ymax, ymin = np.max(ys), np.min(ys)
            staticObstacles.append([xmin, ymin, xmax, ymax])
        rightLaneBoundaries = self.listofObstacles["laneBoundaries"]["rightLaneBoundaries"]
        for boundary in rightLaneBoundaries:
            boundary = translate(boundary, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            boundary = rotate(boundary, angle=inverseEgoAngle, origin=(0, 0))
            xs, ys = boundary.exterior.xy
            xmax, xmin = np.max(xs), np.min(xs)
            ymax, ymin = np.max(ys), np.min(ys)
            staticObstacles.append([xmin, ymin, xmax, ymax])
        frontLaneBoundaries = self.listofObstacles["laneBoundaries"]["frontLaneBoundaries"]
        for boundary in frontLaneBoundaries:
            boundary = translate(boundary, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            boundary = rotate(boundary, angle=inverseEgoAngle, origin=(0, 0))
            xs, ys = boundary.exterior.xy
            xmax, xmin = np.max(xs), np.min(xs)
            ymax, ymin = np.max(ys), np.min(ys)
            staticObstacles.append([xmin, ymin, xmax, ymax])
        rearLaneBoundaries = self.listofObstacles["laneBoundaries"]["rearLaneBoundaries"]
        for boundary in rearLaneBoundaries:
            boundary = translate(boundary, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            boundary = rotate(boundary, angle=inverseEgoAngle, origin=(0, 0))
            xs, ys = boundary.exterior.xy
            xmax, xmin = np.max(xs), np.min(xs)
            ymax, ymin = np.max(ys), np.min(ys)
            staticObstacles.append([xmin, ymin, xmax, ymax])

        # 3. get rid of a selected dynamic obstacle (starting with the ones on-lane)
        dynamicObstaclesXY = list()
        dynamicObstaclesPoly = list()
        dynamicObstaclesUUID = dict()
        riskDict = dict()
        riskFactorRawDict = dict()
        riskFactorDict = dict()
        # allDynamicObstacles = dynamicObstacles["onLaneObstacles"] + dynamicObstacles["offLaneObstacles"]
        # random.shuffle(allDynamicObstacles)
        # allDynamicObstacles = allDynamicObstacles[6:67]

        if self.posUnc == "None":
            print("Positional uncertainty disabled in this frame.")
            allDynamicObstacles = dynamicObstacles["onLaneObstacles"]
            for obstacle in allDynamicObstacles:
                bboxWorldCoord = obstacle.bbox_city_fr
                bboxWorldCoord = bboxWorldCoord[:, 0:2]
                bboxPolygon = geometry.Polygon([bboxWorldCoord[0],
                                                bboxWorldCoord[1],
                                                bboxWorldCoord[3],
                                                bboxWorldCoord[2]])
                bboxPolygon = bboxPolygon.buffer(0.3, cap_style=3)
                bboxPolygon = translate(bboxPolygon, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
                bboxPolygon = rotate(bboxPolygon, angle=inverseEgoAngle, origin=(0, 0))
                xs, ys = bboxPolygon.exterior.xy
                xmax, xmin = np.max(xs), np.min(xs)
                ymax, ymin = np.max(ys), np.min(ys)
                dynamicObstaclesXY.append([xmin, ymin, xmax, ymax])
                dynamicObstaclesUUID[len(dynamicObstaclesXY) - 1] = obstacle.track_uuid
                dynamicObstaclesPoly.append(bboxPolygon)
        elif self.posUnc == "gaussian2DCenter":
            allDynamicObstacles = dynamicObstacles["onLaneObstacles"] + dynamicObstacles["offLaneObstacles"]
            for obstacle in allDynamicObstacles:
                bboxWorldCoord = obstacle.bbox_city_fr
                bboxWorldCoord = bboxWorldCoord[:, 0:2]
                bboxPolygon = geometry.Polygon([bboxWorldCoord[0],
                                                bboxWorldCoord[1],
                                                bboxWorldCoord[3],
                                                bboxWorldCoord[2]])
                centerMeanXY = np.array(self.posUncConfig["centerMeanXY"])
                centerCovariance = np.array(self.posUncConfig["centerCovariance"])
                sampledTrans = np.random.multivariate_normal(centerMeanXY, centerCovariance, 1)[0]
                print(sampledTrans)
                transBboxPolygon = translate(bboxPolygon, xoff=sampledTrans[0], yoff=sampledTrans[1])
                if self._obstacleInLane(self.rasterizedOptimizedLanesAroundTraj, transBboxPolygon):
                    bboxPolygon = transBboxPolygon.buffer(0.3, cap_style=3)
                    bboxPolygon = translate(bboxPolygon, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
                    bboxPolygon = rotate(bboxPolygon, angle=inverseEgoAngle, origin=(0, 0))
                    xs, ys = bboxPolygon.exterior.xy
                    xmax, xmin = np.max(xs), np.min(xs)
                    ymax, ymin = np.max(ys), np.min(ys)
                    dynamicObstaclesXY.append([xmin, ymin, xmax, ymax])
                    dynamicObstaclesUUID[len(dynamicObstaclesXY) - 1] = obstacle.track_uuid
                    dynamicObstaclesPoly.append(bboxPolygon)
                else:
                    print("Obstacle not in lane.")

        elif self.posUnc == "gaussian2DCorners":
            allDynamicObstacles = dynamicObstacles["onLaneObstacles"] + dynamicObstacles["offLaneObstacles"]
            for obstacle in allDynamicObstacles:
                bboxWorldCoord = copy.deepcopy(obstacle.bbox_city_fr)
                bboxWorldCoord = bboxWorldCoord[:, 0:2]
                corner0Mean = np.array(self.posUncConfig["corner0Mean"])
                corner0Covariance = np.array(self.posUncConfig["corner0Covariance"])
                sampledTrans0 = np.random.multivariate_normal(corner0Mean, corner0Covariance, 1)[0]
                bboxWorldCoord[0] += sampledTrans0

                corner1Mean = np.array(self.posUncConfig["corner1Mean"])
                corner1Covariance = np.array(self.posUncConfig["corner1Covariance"])
                sampledTrans1 = np.random.multivariate_normal(corner1Mean, corner1Covariance, 1)[0]
                bboxWorldCoord[1] += sampledTrans1

                corner2Mean = np.array(self.posUncConfig["corner2Mean"])
                corner2Covariance = np.array(self.posUncConfig["corner2Covariance"])
                sampledTrans2 = np.random.multivariate_normal(corner2Mean, corner2Covariance, 1)[0]
                bboxWorldCoord[2] += sampledTrans2

                corner3Mean = np.array(self.posUncConfig["corner3Mean"])
                corner3Covariance = np.array(self.posUncConfig["corner3Covariance"])
                sampledTrans3 = np.random.multivariate_normal(corner3Mean, corner3Covariance, 1)[0]
                bboxWorldCoord[3] += sampledTrans3
                print(sampledTrans0, sampledTrans1, sampledTrans2, sampledTrans3)
                bboxPolygon = geometry.Polygon([bboxWorldCoord[0],
                                                bboxWorldCoord[1],
                                                bboxWorldCoord[3],
                                                bboxWorldCoord[2]])
                if self._obstacleInLane(self.rasterizedOptimizedLanesAroundTraj, bboxPolygon):
                    bboxPolygon = bboxPolygon.buffer(0.3, cap_style=3)
                    bboxPolygon = translate(bboxPolygon, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
                    bboxPolygon = rotate(bboxPolygon, angle=inverseEgoAngle, origin=(0, 0))
                    xs, ys = bboxPolygon.exterior.xy
                    xmax, xmin = np.max(xs), np.min(xs)
                    ymax, ymin = np.max(ys), np.min(ys)
                    dynamicObstaclesXY.append([xmin, ymin, xmax, ymax])
                    dynamicObstaclesUUID[len(dynamicObstaclesXY) - 1] = obstacle.track_uuid
                    dynamicObstaclesPoly.append(bboxPolygon)
                else:
                    print("Obstacle not in lane.")
        else:
            print("Positional Uncertainty not supported.")
            raise NotImplementedError

        # 4. if a goal is reachable available trajectory += 1
        emptySuccess = 0
        for o in range(len(setOfGoals)):
            planner = (self.callPlanner())()
            if self.plannerType == "rrt*":
                initialConditionals = {
                    'start': origin,
                    'end': setOfGoals[o],
                    'obs': np.array(staticObstacles)
                }
                if self.plannerSeed == 0:
                    seed = random.randint(0, 99999)
                    resultX, resultY, success = planner.inference(initialConditionals, seed)
                else:
                    resultX, resultY, success = planner.inference(initialConditionals, self.plannerSeed)
            elif self.plannerType == "hybridA*":
                originYaw = np.append(origin, 0.0)
                setOfGoalsYaw = np.append(setOfGoals[o], 0.0)
                initialConditionals = {
                    'start': originYaw,
                    'end': setOfGoalsYaw,
                    'obs': np.array(staticObstacles)
                }
                resultX, resultY, resultYaw, success = planner.inference(initialConditionals)
            elif self.plannerType == "fot":
                initialConditionals = {
                    'ps': 0,
                    'target_speed': 25,
                    'pos': np.array(origin),
                    'vel': np.array(origin),
                    'wp': np.array([origin, setOfGoals[o]]),
                    'obs': np.array(staticObstacles)
                }
                resultX, resultY, speeds, ix, iy, iyaw, d, s, speedsX, \
                speedsY, misc, costs, success = planner.inference(initialConditionals)
            else:
                raise NotImplementedError
            # self.routingVisualizer(np.array(staticObstacles),
            #                        origin,
            #                        [setOfGoals[o]],
            #                        self.routeVisDir,
            #                        "golden_{}".format(o),
            #                        [],
            #                        [resultX, resultY])
            if success:
                emptySuccess += 1
        if not emptySuccess:
            print("Warning there is no success but we will continue anyway.")

        print("Done evaluating empty grid.")

        fullSuccess = 0
        setOfGoalsLessAllObstacles = []
        for goal in setOfGoals:
            candidate = True
            goalPoint = geometry.Point(goal[0], goal[1])
            for obstacle in dynamicObstaclesPoly:
                if obstacle.exterior.distance(goalPoint) < 1.0:
                    candidate = False
                    break
            if candidate:
                setOfGoalsLessAllObstacles.append(goal)
        for o in range(len(setOfGoalsLessAllObstacles)):
            planner = (self.callPlanner())()
            if self.plannerType == "rrt*":
                initialConditionals = {
                    'start': origin,
                    'end': setOfGoalsLessAllObstacles[o],
                    'obs': np.array(dynamicObstaclesXY + staticObstacles)
                }
                if self.plannerSeed == 0:
                    seed = random.randint(0, 99999)
                    # print("Random seed:", seed)
                    resultX, resultY, success = planner.inference(initialConditionals, seed)
                else:
                    resultX, resultY, success = planner.inference(initialConditionals, self.plannerSeed)
            elif self.plannerType == "hybridA*":
                originYaw = np.append(origin, 0.0)
                setOfGoalsLessAllObstaclesYaw = np.append(setOfGoalsLessAllObstacles[o], 0.0)
                initialConditionals = {
                    'start': originYaw,
                    'end': setOfGoalsLessAllObstaclesYaw,
                    'obs': np.array(dynamicObstaclesXY + staticObstacles)
                }
                resultX, resultY, resultYaw, success = planner.inference(initialConditionals)
            elif self.plannerType == "fot":
                initialConditionals = {
                    'ps': 0,
                    'target_speed': 25,
                    'pos': np.array(origin),
                    'vel': np.array(origin),
                    'wp': np.array([origin, setOfGoalsLessAllObstacles[o]]),
                    'obs': np.array(dynamicObstaclesXY + staticObstacles)
                }
                resultX, resultY, speeds, ix, iy, iyaw, d, s, speedsX, \
                speedsY, misc, costs, success = planner.inference(initialConditionals)
            else:
                raise NotImplementedError
            # self.routingVisualizer(np.array(dynamicObstaclesXY + staticObstacles),
            #                        origin,
            #                        [setOfGoalsLessAllObstacles[o]],
            #                        self.routeVisDir,
            #                        "test_{}".format(o),
            #                        [],
            #                        [resultX, resultY])
            if success:
                fullSuccess += 1
        print("Done evaluating fully occupied grid.")

        for k in range(len(dynamicObstaclesXY)):
            dynamicObstaclesLessOne = copy.deepcopy(dynamicObstaclesXY)
            dynamicObstaclesLessOnePoly = copy.deepcopy(dynamicObstaclesPoly)
            dynamicObstaclesLessOne.pop(k)
            dynamicObstaclesLessOnePoly.pop(k)
            allObstacles = np.array(dynamicObstaclesLessOne + staticObstacles)
            totalSuccess = 0
            setOfGoalsLessObstacles = []
            for goal in setOfGoals:
                candidate = True
                goalPoint = geometry.Point(goal[0], goal[1])
                for obstacle in dynamicObstaclesLessOnePoly:
                    if obstacle.exterior.distance(goalPoint) < 1.0:
                        candidate = False
                        break
                if candidate:
                    setOfGoalsLessObstacles.append(goal)
            for o in range(len(setOfGoalsLessObstacles)):
                planner = (self.callPlanner())()
                if self.plannerType == "rrt*":
                    initialConditionals = {
                        'start': origin,
                        'end': setOfGoalsLessObstacles[o],
                        'obs': allObstacles
                    }
                    if self.plannerSeed == 0:
                        seed = random.randint(0, 99999)
                        # print("Random seed:", seed)
                        resultX, resultY, success = planner.inference(initialConditionals, seed)
                    else:
                        resultX, resultY, success = planner.inference(initialConditionals, self.plannerSeed)
                elif self.plannerType == "hybridA*":
                    originYaw = np.append(origin, 0.0)
                    setOfGoalsLessObstaclesYaw = np.append(setOfGoalsLessObstacles[o], 0.0)
                    initialConditionals = {
                        'start': originYaw,
                        'end': setOfGoalsLessObstaclesYaw,
                        'obs': allObstacles
                    }
                    resultX, resultY, resultYaw, success = planner.inference(initialConditionals)
                elif self.plannerType == "fot":
                    initialConditionals = {
                        'ps': 0,
                        'target_speed': 25,
                        'pos': np.array(origin),
                        'vel': np.array(origin),
                        'wp': np.array([origin, setOfGoalsLessObstacles[o]]),
                        'obs': np.array(allObstacles)
                    }
                    resultX, resultY, speeds, ix, iy, iyaw, d, s, speedsX, \
                    speedsY, misc, costs, success = planner.inference(initialConditionals)
                else:
                    raise NotImplementedError
                # figName = str(timestamp) + "_k" + str(k) + "_o" + str(o)
                # self.routingVisualizer(np.array(dynamicObstaclesXY + staticObstacles),
                #                        origin,
                #                        [setOfGoalsLessObstacles[o]],
                #                        self.routeVisDir,
                #                        figName,
                #                        [k],
                #                        [resultX, resultY])
                if success:
                    totalSuccess += 1
            # 5. calculate the risk based on dynamic obstacles with one remove\
            if emptySuccess:
                riskFactor = (totalSuccess - fullSuccess) * 1.0 / emptySuccess
            else:
                riskFactor = 0.0
            riskDict[str(k)] = totalSuccess
            riskFactorRawDict[str(k)] = riskFactor
            riskFactor = max(riskFactor, 0)
            riskFactorDict[str(k)] = riskFactor
            del dynamicObstaclesLessOne
            del dynamicObstaclesLessOnePoly
            del allObstacles

        riskDict["empty"] = emptySuccess
        riskDict["full"] = fullSuccess
        print(riskFactorRawDict, riskFactorDict, riskDict, dynamicObstaclesUUID)
        if self.visualize:
            self.riskVisualizer(np.array(dynamicObstaclesXY + staticObstacles), riskFactorDict, origin,
                                self.riskVisDir, timestamp)
        print("Done analysing one frame.")
        return riskDict, riskFactorDict, riskFactorRawDict, dynamicObstaclesUUID, dynamicObstaclesXY

    @staticmethod
    def riskVisualizer(obstacles, riskFactorDict, location, savedir, timestamp, visibleRange=80):
        fig, ax = plt.subplots(figsize=(9, 9))
        for idx, obstacle in enumerate(obstacles):
            if str(idx) in riskFactorDict:
                riskColor = riskFactorDict[str(idx)] / (max(riskFactorDict.values()) + 1e-6)
                riskColor = max(riskColor, 0)
                riskColor = (riskColor, 0, 0)
                rect = patch.Rectangle((obstacle[0], obstacle[1]),
                                       obstacle[2] - obstacle[0],
                                       obstacle[3] - obstacle[1], color=riskColor)
            else:
                rect = patch.Rectangle((obstacle[0], obstacle[1]),
                                       obstacle[2] - obstacle[0],
                                       obstacle[3] - obstacle[1], color='blue')
            ax.add_patch(rect)
        plt.plot(location[0], location[1], "vc")
        ax.set_xlim(location[0] - visibleRange, location[0] + visibleRange)
        ax.set_ylim(location[1] - visibleRange, location[1] + visibleRange)
        plt.savefig("{}/{}.jpg".format(savedir, timestamp))
        plt.clf()
        plt.cla()
        plt.close('all')

    @staticmethod
    def routingVisualizer(obstacles, origin, goals, savedir, timestamp, removed=[], routes=None, visibleRange=80):
        # visualize environment before and after planning
        fig, ax = plt.subplots(figsize=(9, 9))
        plt.plot(origin[0], origin[1], "vc")
        for goal in goals:
            plt.plot(goal[0], goal[1], "or")
        if routes is not None and len(routes):
            plt.plot(routes[0], routes[1], ".g", markersize=4)
        for idx, obstacle in enumerate(obstacles):
            if idx not in removed:
                rect = patch.Rectangle((obstacle[0], obstacle[1]),
                                       obstacle[2] - obstacle[0],
                                       obstacle[3] - obstacle[1])
            else:
                rect = patch.Rectangle((obstacle[0], obstacle[1]),
                                       obstacle[2] - obstacle[0],
                                       obstacle[3] - obstacle[1], color="orange", alpha=0.5)
            ax.add_patch(rect)
        ax.set_xlim(origin[0] - visibleRange, origin[0] + visibleRange)
        ax.set_ylim(origin[1] - visibleRange, origin[1] + visibleRange)
        plt.savefig("{}/{}.jpg".format(savedir, timestamp))
        plt.clf()
        plt.cla()
        plt.close('all')


if __name__ == "__main__":
    basePath = path_configs.BASEPATH
    mapDataPath = os.path.join(basePath, "map_files")
    subFolder = sys.argv[1]
    logID = sys.argv[2]
    plannerType = sys.argv[3]
    suffix = sys.argv[4]
    posUnc = sys.argv[5]
    if posUnc == "None":
        prefix = "model_unc"
    elif posUnc in ["gaussian2DCenter", "gaussian2DCorners"]:
        if posUnc == "gaussian2DCenter":
            prefix = "pos_unc_cen"
        if posUnc == "gaussian2DCorners":
            prefix = "pos_unc_cor"
    else:
        raise Exception("Not supported posUnc")
    trajDataPath = os.path.join(basePath, "argoverse-tracking", subFolder)
    routeVisDir = os.path.join(basePath, "visualize_routing_{}".format(plannerType), subFolder, logID, prefix, suffix)
    riskVisDir = os.path.join(basePath, "visualize_risk_{}".format(plannerType), subFolder, logID, prefix, suffix)
    riskSaveDir = os.path.join(basePath, "analysis_risk_{}".format(plannerType), subFolder, logID, prefix)
    if not os.path.exists(routeVisDir):
        os.makedirs(routeVisDir)
    if not os.path.exists(riskVisDir):
        os.makedirs(riskVisDir)
    if not os.path.exists(riskSaveDir):
        try:
            os.makedirs(riskSaveDir)
        except FileExistsError:
            print("Folder exist continue...")
    if len(sys.argv) == 6:
        genRisk = GenerateRisk(mapDataPath=mapDataPath,
                               trajDataPath=trajDataPath,
                               logID=logID,
                               plannerType=plannerType,
                               routeVisDir=routeVisDir,
                               riskVisDir=riskVisDir,
                               riskSaveDir=riskSaveDir,
                               visualize=True,
                               posUnc=posUnc)
    elif len(sys.argv) > 6:
        seed = sys.argv[6]
        genRisk = GenerateRisk(mapDataPath=mapDataPath,
                               trajDataPath=trajDataPath,
                               logID=logID,
                               plannerType=plannerType,
                               routeVisDir=routeVisDir,
                               riskVisDir=riskVisDir,
                               riskSaveDir=riskSaveDir,
                               plannerSeed=int(seed),
                               visualize=False,
                               posUnc=posUnc)
    else:
        raise ValueError
    genRisk.riskAnalyser(saveFileName="{}_risk_analysis.pkl".format(suffix))
