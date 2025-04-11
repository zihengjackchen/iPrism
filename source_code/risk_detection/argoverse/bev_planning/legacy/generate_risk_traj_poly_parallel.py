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

from ast import arguments
import copy
import os
import time
from typing import List
from cv2 import mean

from concurrent.futures import ProcessPoolExecutor
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np
import json
import gc
import sys
import random
import pickle as pkl
import logging

from frenet_hyperparameters import STATIC_FOT_HYPERPARAMETERS
from argoverse.utils.interpolate import interp_arc
from shapely import geometry
from shapely.affinity import translate, rotate
from scipy.spatial.transform import Rotation as R
from planners import RRTStarPlanner, HybridAStarPlanner, FOTPlanner

from generate_bev import GenerateBEVGraph

logging.basicConfig(level=logging.ERROR)
mpl.use('Agg')
print(mpl.get_backend())


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
            with open("./positional_uncertainty.json") as f:
                self.posUncConfig = json.load(f)
            assert self.posUnc in self.posUncConfig
            self.posUncConfig = self.posUncConfig[self.posUnc]
            print("Uncertainty parameters:", self.posUncConfig)
        else:
            print("Positional uncertainty is not enabled for this run.")
        if self.plannerSeed:
            np.random.seed(self.plannerSeed)

    def riskAnalyser(self, saveFileName=None, lookForwardDist=120, lookForwardTime=3):
        if self.riskSaveDir and saveFileName:
            if os.path.isfile(os.path.join(self.riskSaveDir, saveFileName)):
                print("{} exists, already analysed, return...".format(os.path.join(self.riskSaveDir, saveFileName)))
                return
        bookKeepingPerScene = dict()
        for timestamp in self.bevGraph.listofTimestamps:
            riskDict, riskFactorDict, riskFactorRawDict, \
                dynamicObstaclesUUID, dynamicObstaclesPoly = self._riskAnalyserOneTimestamp(timestamp, lookForwardDist, lookForwardTime)
            bookKeepingPerScene[timestamp] = {
                "riskDict": riskDict,
                "riskFactorDict": riskFactorDict,
                "riskFactorRawDict": riskFactorRawDict,
                "dynamicObstaclesUUID": dynamicObstaclesUUID,
                "dynamicObstaclesPoly": dynamicObstaclesPoly
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
    def _getTimeBasedObstacles(listOfTime, staticObstaclesPoly, dynamicObstaclesPoly, dynamicObstaclesUUID, remove=[]):
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
    
    def _fot_inference_wrapper(self, funcArguments):
        planner = (self.callPlanner())()
        fotInitialState = funcArguments["fotInitialState"]
        referenceWaypoints = funcArguments["referenceWaypoints"]
        timeBasedObstacles = funcArguments["timeBasedObstacles"]
        fotHyperparameters = funcArguments["fotHyperparameters"]
        visualizeRoute = funcArguments["visualizeRoute"]
        origin = funcArguments["origin"]
        visualizeSaveName = funcArguments["visualizeSaveName"]
        removedList = funcArguments["removedList"]

        plannedTraj, refTraj, success = planner.inference(fotInitialState, referenceWaypoints,
                                                          timeBasedObstacles, fotHyperparameters)
        resultX = plannedTraj.x
        resultY = plannedTraj.y
        if visualizeRoute:
            if len(removedList):
                timeBasedObstaclesVis = funcArguments["timeBasedObstaclesVis"]
                timeBasedObstacles = timeBasedObstaclesVis
            self.routingVisualizer(timeBasedObstacles,
                                   origin,
                                   [referenceWaypoints],
                                   self.routeVisDir,
                                   visualizeSaveName,
                                   removedList,
                                   [resultX, resultY],
                                   refTraj)
        return success

    # TODO we could get rid of the min distance as 5 because it cannot be reached
    # anyway if it is too close to the ego vehicle, and add a parameter T to specify
    # the look forward time
    def _riskAnalyserOneTimestamp(self, timestamp, lookForwardDist, lookForwardTime,
                                  minDist=5, distanceToStaOb=0.1, distanceToDynOb=1.5):
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
            return {}, {}, {}, [], []

        # taka list of dynamic obstacles that aligns with the time stamps
        dynamicObstacles = list()
        for t in listOfTime:
            dynamicObstacles.append(self.listofObstacles["dynamicObstacles"][t])

        # 1. find the next set of POI based on the current location
        setOfGoals = []
        for laneID in self.rasterizedOptimizedLanesAroundTraj["uniqueLaneIDs"]:
            lanePOI = copy.deepcopy(self.rasterizedOptimizedLanesAroundTraj["processedLane"][laneID]["lanePOI"])
            lanePOIOrin = copy.deepcopy(lanePOI)
            if len(lanePOI):
                lanePOI[:, 0] = lanePOI[:, 0] + inverseEgoTranslate[0]
                lanePOI[:, 1] = lanePOI[:, 1] + inverseEgoTranslate[1]
                lanePOI = lanePOI @ inverseEgoRotate[:2, :2].T
                for idx, POI in enumerate(lanePOI):
                    # TODO: think about whether we can user all point of interest instead of imposing a max distance
                    if minDist < POI[0] <= lookForwardDist:
                        POI = self._convertPOItoRefPath(poseCitytoEgo["translation"][0:2], lanePOIOrin[idx][0:2])
                        POI = np.array(POI)
                        POI[:, 0] = POI[:, 0] + inverseEgoTranslate[0]
                        POI[:, 1] = POI[:, 1] + inverseEgoTranslate[1]
                        POI = POI @ inverseEgoRotate[:2, :2].T
                        setOfGoals.append(list(POI))

        # 2. origin as current position, list of static boundary obstacles, and goal as POI
        setOfGoals = sorted(setOfGoals, key=lambda element: (element[3][0]))
        # if len(setOfGoals) > 20:
        #     print("Restricting the number of goals to 20, currently the length is {}.".format(len(setOfGoals)))
        #     setOfGoals = setOfGoals[:20]
        setOfGoals = np.array(setOfGoals)
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
            elif self.posUnc == "gaussian2DShift":  # small displacement
                allDynamicObstacles = dynamicObstacles[tdx]["onLaneObstacles"] + dynamicObstacles[tdx][
                    "offLaneObstacles"]
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
                    transBboxPolygon = translate(bboxPolygon, xoff=sampledTrans[0], yoff=sampledTrans[1])
                    if self._obstacleInLane(self.rasterizedOptimizedLanesAroundTraj, transBboxPolygon):
                        bboxPolygon = translate(transBboxPolygon, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
                        bboxPolygon = rotate(bboxPolygon, angle=inverseEgoAngle, origin=(0, 0))
                        currentDynamicObstaclesPolyOrin.append(bboxPolygon)
                        bboxPolygon = bboxPolygon.buffer(distanceToDynOb, cap_style=3)
                        currentDynamicObstaclesPoly.append(bboxPolygon)
                        currentDynamicObstaclesUUID[len(currentDynamicObstaclesPoly) - 1] = obstacle.track_uuid
                    else:
                        logging.info("Obstacle not in lane.")
            elif self.posUnc == "gaussian2DRotate":  # small rotation
                allDynamicObstacles = dynamicObstacles[tdx]["onLaneObstacles"] + dynamicObstacles[tdx][
                    "offLaneObstacles"]
                for obstacle in allDynamicObstacles:
                    bboxWorldCoord = obstacle.bbox_city_fr
                    bboxWorldCoord = bboxWorldCoord[:, 0:2]
                    bboxPolygon = geometry.Polygon([bboxWorldCoord[0],
                                                    bboxWorldCoord[1],
                                                    bboxWorldCoord[3],
                                                    bboxWorldCoord[2]])
                    rotationMean = np.array(self.posUncConfig["rotationMean"])
                    rotationVariance = np.array(self.posUncConfig["rotationVariance"])
                    sampledRota = np.random.normal(rotationMean, np.sqrt(rotationVariance), 1)[0]
                    rotaBboxPolygon = rotate(bboxPolygon, angle=sampledRota)
                    if self._obstacleInLane(self.rasterizedOptimizedLanesAroundTraj, rotaBboxPolygon):
                        bboxPolygon = translate(rotaBboxPolygon, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
                        bboxPolygon = rotate(bboxPolygon, angle=inverseEgoAngle, origin=(0, 0))
                        currentDynamicObstaclesPolyOrin.append(bboxPolygon)
                        bboxPolygon = bboxPolygon.buffer(distanceToDynOb, cap_style=3)
                        currentDynamicObstaclesPoly.append(bboxPolygon)
                        currentDynamicObstaclesUUID[len(currentDynamicObstaclesPoly) - 1] = obstacle.track_uuid
                    else:
                        logging.info("Obstacle not in lane.")
            elif self.posUnc == "gaussian2DCorners":  # deformation
                allDynamicObstacles = dynamicObstacles[tdx]["onLaneObstacles"] + dynamicObstacles[tdx][
                    "offLaneObstacles"]
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
                        bboxPolygon = translate(bboxPolygon, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
                        bboxPolygon = rotate(bboxPolygon, angle=inverseEgoAngle, origin=(0, 0))
                        currentDynamicObstaclesPolyOrin.append(bboxPolygon)
                        bboxPolygon = bboxPolygon.buffer(distanceToDynOb, cap_style=3)
                        currentDynamicObstaclesPoly.append(bboxPolygon)
                        currentDynamicObstaclesUUID[len(currentDynamicObstaclesPoly) - 1] = obstacle.track_uuid
                    else:
                        logging.info("Obstacle not in lane.")
            elif self.posUnc == "gaussian2DShiftRotate":  # deformation
                allDynamicObstacles = dynamicObstacles[tdx]["onLaneObstacles"] + dynamicObstacles[tdx][
                    "offLaneObstacles"]
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
                    transBboxPolygon = translate(bboxPolygon, xoff=sampledTrans[0], yoff=sampledTrans[1])
                    rotationMean = np.array(self.posUncConfig["rotationMean"])
                    rotationVariance = np.array(self.posUncConfig["rotationVariance"])
                    sampledRota = np.random.normal(rotationMean, np.sqrt(rotationVariance), 1)[0]
                    rotaBboxPolygon = rotate(transBboxPolygon, angle=sampledRota)
                    if self._obstacleInLane(self.rasterizedOptimizedLanesAroundTraj, rotaBboxPolygon):
                        bboxPolygon = translate(rotaBboxPolygon, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
                        bboxPolygon = rotate(bboxPolygon, angle=inverseEgoAngle, origin=(0, 0))
                        currentDynamicObstaclesPolyOrin.append(bboxPolygon)
                        bboxPolygon = bboxPolygon.buffer(distanceToDynOb, cap_style=3)
                        currentDynamicObstaclesPoly.append(bboxPolygon)
                        currentDynamicObstaclesUUID[len(currentDynamicObstaclesPoly) - 1] = obstacle.track_uuid
                    else:
                        logging.info("Obstacle not in lane.")
            else:
                logging.error("Uncertainty type not supported.")
                raise NotImplementedError
            # dynamicObstaclesXY.append(currentDynamicObstaclesXY)
            dynamicObstaclesPoly.append(currentDynamicObstaclesPoly)
            dynamicObstaclesUUID.append(currentDynamicObstaclesUUID)
            dynamicObstaclesPolyOrin.append(currentDynamicObstaclesPolyOrin)

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

        fotHyperparameters = STATIC_FOT_HYPERPARAMETERS

        # dynamic hyperparameters depending on speed
        fotHyperparameters["DT"] = averageTimeTick
        fotHyperparameters["MAX_T"] = lookForwardTime + averageTimeTick + 0.00001
        fotHyperparameters["MIN_T"] = lookForwardTime + averageTimeTick
        fotHyperparameters["TARGET_SPEED"] = fotHyperparameters["MIN_T"] * fotHyperparameters["MAX_ACCEL"] * 0.50 + initialSpeed
        fotHyperparameters["D_T_S"] = abs(fotHyperparameters["TARGET_SPEED"] - initialSpeed) * 0.5 / fotHyperparameters["N_S_SAMPLE"]
        print(fotHyperparameters["DT"], fotHyperparameters["MIN_T"], fotHyperparameters["TARGET_SPEED"],
              fotHyperparameters["D_T_S"], fotHyperparameters["D_T_S"]*fotHyperparameters["N_S_SAMPLE"])
        fotInitialState = {
            "c_speed": initialSpeed,
            "c_d": 0,
            "c_d_d": 0,
            "c_d_dd": 0,
            "s0": 0
        }
        # purge theoratically unreachable goals
        print("Before purging:", len(setOfGoals))
        reachableGoal = []
        # theoryReachable = (fotHyperparameters["TARGET_SPEED"] + fotHyperparameters["D_T_S"] * fotHyperparameters["N_S_SAMPLE"]) * fotHyperparameters["MIN_T"]
        terminalSpeed = (fotHyperparameters["TARGET_SPEED"] + fotHyperparameters["D_T_S"] * fotHyperparameters["N_S_SAMPLE"])
        theoryReachable = ((terminalSpeed + initialSpeed) / 2) * fotHyperparameters["MIN_T"]
        for waypoints in setOfGoals:
            dest = waypoints[-1]
            if np.linalg.norm(dest - origin) < theoryReachable:
                reachableGoal.append(waypoints)
        setOfGoals = reachableGoal
        print("After purging:", len(setOfGoals))

        # 4. check if the goal is reachable or not with removing one obstacle per time
        emptySuccess = 0
        successGoals = list()
        timeBasedObstacles = self._getTimeBasedObstacles(listOfTime, staticObstaclesPoly,
                                                         None, None)
        # prepare arguments for parallel execution
        emptyArguments = list()
        for o in range(len(setOfGoals)):
            emptyArgument = dict()
            referenceWaypoints = setOfGoals[o]
            emptyArgument["fotInitialState"] = fotInitialState
            emptyArgument["referenceWaypoints"] = referenceWaypoints
            emptyArgument["timeBasedObstacles"] = timeBasedObstacles
            emptyArgument["fotHyperparameters"] = fotHyperparameters
            emptyArgument["visualizeRoute"] = False
            emptyArgument["origin"] = origin
            emptyArgument["visualizeSaveName"] = "{}_golden_{}".format(timestamp, o)
            emptyArgument["removedList"] = []
            emptyArguments.append(emptyArgument)
        with ProcessPoolExecutor() as executor:
            successResults = executor.map(self._fot_inference_wrapper, emptyArguments)
        for o, success in enumerate(successResults):
            if success:
                emptySuccess += 1
                successGoals.append(setOfGoals[o])
        if not emptySuccess:
            logging.warning("Warning there is no success but we will continue anyway.")
        print("Done evaluating empty grid, empty success:", emptySuccess, "succeed goals:", successGoals)

        fullSuccess = 0
        successGoals = list()
        setOfGoalsAllObstacles = setOfGoals
        timeBasedObstacles = self._getTimeBasedObstacles(listOfTime, staticObstaclesPoly,
                                                         dynamicObstaclesPoly, dynamicObstaclesUUID)
        # prepare arguments for parallel execution
        fullArguments = list()
        for o in range(len(setOfGoalsAllObstacles)):
            fullArgument = dict()
            referenceWaypoints = setOfGoalsAllObstacles[o]
            fullArgument["fotInitialState"] = fotInitialState
            fullArgument["referenceWaypoints"] = referenceWaypoints
            fullArgument["timeBasedObstacles"] = timeBasedObstacles
            fullArgument["fotHyperparameters"] = fotHyperparameters
            fullArgument["visualizeRoute"] = False
            fullArgument["origin"] = origin
            fullArgument["visualizeSaveName"] = "{}_full_{}".format(timestamp, o)
            fullArgument["removedList"] = []
            fullArguments.append(fullArgument)
        with ProcessPoolExecutor() as executor:
            successResults = executor.map(self._fot_inference_wrapper, fullArguments)
        for o, success in enumerate(successResults):
            if success:
                fullSuccess += 1
                successGoals.append(setOfGoalsAllObstacles[o])
        print("Done evaluating fully occupied grid, full obstacle success:", fullSuccess, "succeed goals:",
              successGoals)

        # 5. Finally remove obstacles one by one
        uuidCurrentTime = list(dynamicObstaclesUUID[0].values())
        for k in range(len(uuidCurrentTime)):
            dynamicObstaclesLessOnePoly = copy.deepcopy(dynamicObstaclesPoly)
            uuid = uuidCurrentTime[k]
            print("UUID to be removed at time {} is {}.".format(timestamp, uuid))
            lessOneSuccess = 0
            setOfGoalsLessOneObstacles = setOfGoals
            timeBasedObstaclesLessOne = self._getTimeBasedObstacles(listOfTime, 
                                                                    staticObstaclesPoly,
                                                                    dynamicObstaclesLessOnePoly, 
                                                                    dynamicObstaclesUUID,
                                                                    [uuid])
            # prepare arguments for parallel execution
            lessOneArguments = list()
            for o in range(len(setOfGoalsLessOneObstacles)):
                lessOneArgument = dict()
                referenceWaypoints = setOfGoalsLessOneObstacles[o]
                lessOneArgument["fotInitialState"] = fotInitialState
                lessOneArgument["referenceWaypoints"] = referenceWaypoints
                lessOneArgument["timeBasedObstacles"] = timeBasedObstaclesLessOne
                lessOneArgument["timeBasedObstaclesVis"] = timeBasedObstacles
                lessOneArgument["fotHyperparameters"] = fotHyperparameters
                lessOneArgument["visualizeRoute"] = False
                lessOneArgument["origin"] = origin
                lessOneArgument["visualizeSaveName"] = "{}_lessone_{}_{}".format(timestamp, k, o)
                lessOneArgument["removedList"] = [uuid]
                lessOneArguments.append(lessOneArgument)
            with ProcessPoolExecutor() as executor:
                successResults = executor.map(self._fot_inference_wrapper, lessOneArguments)
            for o, success in enumerate(successResults):
                if success:
                    lessOneSuccess += 1
            if emptySuccess:
                riskFactor = (lessOneSuccess - fullSuccess) * 1.0 / emptySuccess
            else:
                riskFactor = 0.0
            riskDict[str(k)] = lessOneSuccess
            riskFactorRawDict[str(k)] = riskFactor
            riskFactor = max(riskFactor, 0)
            riskFactorDict[str(k)] = riskFactor
            del dynamicObstaclesLessOnePoly

        riskDict["empty"] = emptySuccess
        riskDict["full"] = fullSuccess
        print("Done analysing one frame after {} seconds.".format(time.time() - startTime))
        print(riskFactorRawDict, riskFactorDict, riskDict, dynamicObstaclesUUID[0])
        if self.visualize:
            self.riskVisualizer(np.array(dynamicObstaclesPolyOrin[0] + staticObstaclesPolyOrin),
                                riskFactorDict, origin, self.riskVisDir, timestamp)
        return riskDict, riskFactorDict, riskFactorRawDict, dynamicObstaclesUUID, dynamicObstaclesPoly

    @staticmethod
    def riskVisualizer(obstacles, riskFactorDict, location, savedir, timestamp, visibleRange=80):
        fig, ax = plt.subplots(figsize=(9, 9))
        for idx, obstacle in enumerate(obstacles):
            if str(idx) in riskFactorDict:
                riskColor = riskFactorDict[str(idx)] / (max(riskFactorDict.values()) + 1e-6)
                riskColor = max(riskColor, 0)
                riskColor = (riskColor, 0, 0)
                xs, ys = obstacle.exterior.xy
                plt.fill(xs, ys, alpha=1.0, fc=riskColor, ec=riskColor)
            else:
                xs, ys = obstacle.exterior.xy
                plt.fill(xs, ys, alpha=1.0, fc="blue", ec="blue")
        plt.plot(location[0], location[1], "vc")
        ax.set_xlim(location[0] - visibleRange, location[0] + visibleRange)
        ax.set_ylim(location[1] - visibleRange, location[1] + visibleRange)
        plt.savefig("{}/{}.jpg".format(savedir, timestamp))
        plt.close()

    def routingVisualizer(self, obstacles, origin, goals, savedir, savename, removed=[],
                          routes=None, refRoutes=None, visibleRange=80):
        # visualize environment before and after planning
        fig, ax = plt.subplots(figsize=(9, 9))
        plt.plot(origin[0], origin[1], "vc")
        for goal in goals:
            plt.plot(goal[-1][0], goal[-1][1], "or")
        if routes is not None and len(routes):
            plt.plot(routes[0], routes[1], ".b")
        if refRoutes is not None and len(refRoutes[0]) and len(refRoutes[1]):
            plt.plot(refRoutes[0], refRoutes[1], "-g")
        for t_obs in obstacles["static"]:
            for idx, obstacle in enumerate(t_obs):
                xs, ys = obstacle.exterior.xy
                plt.fill(xs, ys, alpha=0.9, fc='blue', ec='blue')
        for odx in range(len(obstacles["dynamic"])):
            for uuid, obstacle in obstacles["dynamic"][odx].items():
                xs, ys = obstacle.exterior.xy
                if uuid not in removed:
                    plt.fill(xs, ys, alpha=((odx + 1) / len(obstacles["dynamic"])), fc='orange', ec='orange')
                else:
                    plt.fill(xs, ys, alpha=((odx + 1) / len(obstacles["dynamic"])), fc='royalblue', ec='royalblue')
        ax.set_xlim(origin[0] - visibleRange, origin[0] + visibleRange)
        ax.set_ylim(origin[1] - visibleRange, origin[1] + visibleRange)
        plt.savefig("{}/{}.jpg".format(savedir, savename))
        plt.close()


if __name__ == "__main__":
    basePath = path_configs.BASEPATH
    mapDataPath = os.path.join(basePath, "map_files")
    subFolder = sys.argv[1]
    logID = sys.argv[2]
    plannerType = sys.argv[3]
    assert plannerType == "fot*", "Planner type for trajectory based risk analysis can only be fot*"
    suffix = sys.argv[4]
    posUnc = sys.argv[5]
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
                               visualize=True,
                               posUnc=posUnc)
    else:
        raise ValueError
    genRisk.riskAnalyser(saveFileName="{}_risk_analysis.pkl".format(suffix))
