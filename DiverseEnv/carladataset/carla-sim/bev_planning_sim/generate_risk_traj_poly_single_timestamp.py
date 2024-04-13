"""
MIT License

Copyright (c) 2024 submission #104

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

import os
import time
import path_configs
from concurrent.futures import ThreadPoolExecutor
import matplotlib as mpl
mpl.use('Agg')
print(mpl.get_backend())
import matplotlib.pyplot as plt
import json
import sys
import pickle as pkl
import logging

from frenet_hyperparameters import STATIC_FOT_HYPERPARAMETERS
from planners import RRTStarPlanner, HybridAStarPlanner, FOTPlanner
from motion_prediction import *


logging.basicConfig(level=logging.ERROR)


class GenerateRiskCarla:
    def __init__(self, trajDataPath, logID, plannerType, suffix, 
                 routeVisDir, riskVisDir, riskSaveDir, concurrentObjCount,
                 plannerSeed=0, visualize=False, posUnc="None", prediction='GT'):
        self.pklDataPath = os.path.join(trajDataPath, logID)
        f = open(self.pklDataPath, "rb")
        data_dict = pkl.load(f)
        f.close()
        self.rasterizedOptimizedLanesAroundTraj = data_dict["rasterizedOptimizedLanesAroundTraj"]
        self.listofObstacles = data_dict["snapshotObstacles"]
        self.listofTimestamps = data_dict["listOfTimestamp"]
        # down sampling timestamps
        self.listofTimestamps = [self.listofTimestamps[t] for t in range(0, len(self.listofTimestamps), 4)]
        self.egoTelemetry = data_dict["egoTelemetry"]
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
        self.prediction = prediction
        print("Prediction type:", self.prediction)

    def riskAnalyserSingleTimestamp(self, timestamp, timestampIdx, saveFileName=None, 
                                    lookForwardDist=120, lookForwardTime=3):
        bookKeepingPerScene = dict()
        if timestamp in self.listofTimestamps:
            raArguement = dict()
            raArguement["timestamp"] = timestamp
            raArguement["lookForwardDist"] = lookForwardDist
            raArguement["lookForwardTime"] = lookForwardTime
            raArguement["timestampIdx"] = timestampIdx
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
            print("Saved analysis pkl file to:", os.path.join(self.riskSaveDir, "tmp_{}".format(self.suffix), saveFileName))

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

    def _getInitialSpeed(self, listOfTime):
        trans1 = self.egoTelemetry[listOfTime[1]]["location"][0:2]
        trans2 = self.egoTelemetry[listOfTime[0]]["location"][0:2]
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

    def _convertPOItoRefPath(self, egoPosition, egoPositionNext, backupNext, POI, prevPOI, travelDistFrac=0.1):
        egoLaneVector = egoPositionNext - egoPosition
        if np.linalg.norm(egoLaneVector) < 0.001:
            print("Using waypoint as next direction.")
            egoLaneVector = backupNext - egoPosition
        unitEgoLaneVector = egoLaneVector / (np.linalg.norm(egoLaneVector))
        egoToDest = POI[0:2] - egoPosition[0:2]
        distToDest = np.dot(unitEgoLaneVector, egoToDest)
        scaledEgoLaneVector = unitEgoLaneVector * distToDest * travelDistFrac

        poiLaneVector = prevPOI - POI
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
        timeBasedObstaclesOrin = funcArguments["timeBasedObstaclesOrin"]
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
                timeBasedObstaclesOrin = timeBasedObstaclesVis
            self.routingVisualizer(timeBasedObstaclesOrin,
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
    def _riskAnalyserOneTimestamp(self, raArguement):
        timestampIdx = raArguement["timestampIdx"]
        timestamp = raArguement["timestamp"]
        lookForwardDist = raArguement["lookForwardDist"]
        lookForwardTime = raArguement["lookForwardTime"]
        minDist = 5
        distanceToStaOb = 0.1
        distanceToDynOb = 1.5
        print("Start Processing Time stamp: {}".format(timestamp))
        startTime = time.time()
        # 0. get essential information for translation and rotation and a list of relavant timestamps
        poseCitytoEgo = self.egoTelemetry[timestamp]
        inverseEgoTranslate = -np.array(poseCitytoEgo["location"][0:2])
        inverseEgoRotate = -poseCitytoEgo["rotation"]
        inverseEgoAngle = inverseEgoRotate[1]  # Inverse Yaw
        listOfTime = [timestamp]
        for t in self.listofTimestamps:
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

        # choose to use ground truth data
        if self.prediction == "GT":
            for t in listOfTime:
                dynamicObstacles.append(self.listofObstacles["dynamicObstacles"][t])
        # choose to use motion prediction model
        elif self.prediction == "CVCTR":
            for tdx in range(len(listOfTime)):
                if tdx == 0:
                    dynamicObstacles.append(self.listofObstacles["dynamicObstacles"][listOfTime[tdx]])
                else:
                    prev1TimeObstacleDict = dynamicObstacles[-1]
                    if len(dynamicObstacles) > 1:  # if we have more than two datapoints already
                        prev2TimeObstacleDict = dynamicObstacles[-2]
                        previousTime2 = listOfTime[tdx-2]
                    elif len(dynamicObstacles) <= 1 and timestampIdx > 0:  # if we only have one datapoint
                        prev2TimeObstacleDict = self.listofObstacles["dynamicObstacles"][self.listofTimestamps[timestampIdx-1]]
                        previousTime2 = self.listofTimestamps[timestampIdx-1]
                    else:  # if there is no prior GT data, meaning that this is the first frame, use default profile
                        prev2TimeObstacleDict = dict()
                        previousTime2 = -1  # invalid
                    predictedObstacleDict = cvctrPrediction(prev1TimeObstacleDict,
                                                            prev2TimeObstacleDict,
                                                            listOfTime[tdx],
                                                            listOfTime[tdx-1],
                                                            previousTime2)
                    dynamicObstacles.append(predictedObstacleDict)
        else:
            raise NotImplementedError

        # Debugging
        # for idx, o in enumerate(dynamicObstacles):
        #     obstacle = o["onLaneObstacles"][0]
        #     coord0 = obstacle["bboxWorldVertices"][0][0:2]
        #     coord1 = obstacle["bboxWorldVertices"][2][0:2]
        #     coord2 = obstacle["bboxWorldVertices"][6][0:2]
        #     coord3 = obstacle["bboxWorldVertices"][4][0:2]
        #     worldBboxPolygon = geometry.Polygon([coord0, coord1, coord2, coord3])
        #     xs, ys = worldBboxPolygon.exterior.xy
        #     plt.fill(xs, ys, fc='red', ec='red', alpha=(idx+1)/len(dynamicObstacles))
        #     plt.scatter(obstacle['location'][0], obstacle['location'][1], c="blue")
        # plt.show()
        # raise NotImplementedError

        # 1. find the next set of POI based on the current location
        setOfGoals = []
        for laneID in self.rasterizedOptimizedLanesAroundTraj["uniqueLaneIDs"]:
            lanePOI = copy.deepcopy(self.rasterizedOptimizedLanesAroundTraj["processedLane"][laneID]["lanePOI"])[:, 0:2]
            lanePOIOrin = copy.deepcopy(lanePOI)
            lanePOIOrinPrev = copy.deepcopy(self.rasterizedOptimizedLanesAroundTraj["processedLane"][laneID]["lanePOIPrev"])[:, 0:2]
            if len(lanePOI):
                lanePOI[:, 0] = lanePOI[:, 0] + inverseEgoTranslate[0]
                lanePOI[:, 1] = lanePOI[:, 1] + inverseEgoTranslate[1]
                inverseRotationMatrix = np.array([
                    [np.cos(np.deg2rad(inverseEgoAngle)), -np.sin(np.deg2rad(inverseEgoAngle))],
                    [np.sin(np.deg2rad(inverseEgoAngle)), np.cos(np.deg2rad(inverseEgoAngle))]
                ])
                lanePOI = lanePOI @ inverseRotationMatrix.T
                for idx, POI in enumerate(lanePOI):
                    # TODO: think about whether we can user all point of interest instead of imposing a max distance
                    if minDist < POI[0] <= lookForwardDist:
                        POI = self._convertPOItoRefPath(poseCitytoEgo["location"][0:2],
                                                        self.egoTelemetry[self.listofTimestamps[timestampIdx+1]]["location"][0:2],
                                                        poseCitytoEgo["next"][0:2],
                                                        lanePOIOrin[idx][0:2],
                                                        lanePOIOrinPrev[idx][0:2])
                        POI = np.array(POI)
                        POI[:, 0] = POI[:, 0] + inverseEgoTranslate[0]
                        POI[:, 1] = POI[:, 1] + inverseEgoTranslate[1]
                        POI = POI @ inverseRotationMatrix.T
                        setOfGoals.append(list(POI))

        # 2. origin as current position, list of static boundary obstacles, and goal as POI
        setOfGoals = sorted(setOfGoals, key=lambda element: (element[3][0]))
        setOfGoals = np.array(setOfGoals)
        origin = np.array([0, 0])
        staticObstaclesPoly = []
        staticObstaclesPolyOrin = []
        leftLaneBoundaries = self.rasterizedOptimizedLanesAroundTraj["processedLane"][0]["polygonLeftLaneBound"]
        for boundary in leftLaneBoundaries:
            boundary = translate(boundary, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            boundary = rotate(boundary, angle=inverseEgoAngle, origin=(0, 0))
            staticObstaclesPolyOrin.append(boundary)
            boundary = boundary.buffer(distanceToStaOb, cap_style=3)
            staticObstaclesPoly.append(boundary)
        rightLaneBoundaries = self.rasterizedOptimizedLanesAroundTraj["processedLane"][0]["polygonRightLaneBound"]
        for boundary in rightLaneBoundaries:
            boundary = translate(boundary, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            boundary = rotate(boundary, angle=inverseEgoAngle, origin=(0, 0))
            staticObstaclesPolyOrin.append(boundary)
            boundary = boundary.buffer(distanceToStaOb, cap_style=3)
            staticObstaclesPoly.append(boundary)
        stateicObstacles = self.listofObstacles["staticObstacles"]
        for sob in stateicObstacles:
            world_bbox_coord = sob["bboxWorldVertices"]
            coord0 = world_bbox_coord[0][0:2]
            coord1 = world_bbox_coord[2][0:2]
            coord2 = world_bbox_coord[6][0:2]
            coord3 = world_bbox_coord[4][0:2]
            static_obs_bbox_poly = geometry.Polygon([coord0, coord1, coord2, coord3])
            static_obs_bbox_poly = translate(static_obs_bbox_poly, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
            static_obs_bbox_poly = rotate(static_obs_bbox_poly, angle=inverseEgoAngle, origin=(0, 0))
            staticObstaclesPolyOrin.append(static_obs_bbox_poly)
            static_obs_bbox_poly = static_obs_bbox_poly.buffer(distanceToStaOb, cap_style=3)
            staticObstaclesPoly.append(static_obs_bbox_poly)

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
                    bboxWorldCoord = obstacle["bboxWorldVertices"]
                    coord0 = bboxWorldCoord[0][0:2]
                    coord1 = bboxWorldCoord[2][0:2]
                    coord2 = bboxWorldCoord[6][0:2]
                    coord3 = bboxWorldCoord[4][0:2]
                    bboxPolygon = geometry.Polygon([coord0, coord1, coord2, coord3])

                    # we do not need to buffer it anymore, the frenet will take care of the buffering
                    bboxPolygon = translate(bboxPolygon, xoff=inverseEgoTranslate[0], yoff=inverseEgoTranslate[1])
                    bboxPolygon = rotate(bboxPolygon, angle=inverseEgoAngle, origin=(0, 0))
                    currentDynamicObstaclesPolyOrin.append(bboxPolygon)
                    bboxPolygon = bboxPolygon.buffer(distanceToDynOb, cap_style=3)
                    currentDynamicObstaclesPoly.append(bboxPolygon)
                    currentDynamicObstaclesUUID[len(currentDynamicObstaclesPoly) - 1] = obstacle["Id"]
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
                        bboxPolygon = translate(transBboxPolygon, xoff=inverseEgoTranslate[0],
                                                yoff=inverseEgoTranslate[1])
                        bboxPolygon = rotate(bboxPolygon, angle=inverseEgoAngle, origin=(0, 0))
                        currentDynamicObstaclesPolyOrin.append(bboxPolygon)
                        bboxPolygon = bboxPolygon.buffer(distanceToDynOb, cap_style=3)
                        currentDynamicObstaclesPoly.append(bboxPolygon)
                        currentDynamicObstaclesUUID[len(currentDynamicObstaclesPoly) - 1] = obstacle["Id"]
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
                        bboxPolygon = translate(rotaBboxPolygon, xoff=inverseEgoTranslate[0],
                                                yoff=inverseEgoTranslate[1])
                        bboxPolygon = rotate(bboxPolygon, angle=inverseEgoAngle, origin=(0, 0))
                        currentDynamicObstaclesPolyOrin.append(bboxPolygon)
                        bboxPolygon = bboxPolygon.buffer(distanceToDynOb, cap_style=3)
                        currentDynamicObstaclesPoly.append(bboxPolygon)
                        currentDynamicObstaclesPoly.append(bboxPolygon)
                        currentDynamicObstaclesUUID[len(currentDynamicObstaclesPoly) - 1] = obstacle["Id"]
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
                        currentDynamicObstaclesUUID[len(currentDynamicObstaclesPoly) - 1] = obstacle["Id"]
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
                        bboxPolygon = translate(rotaBboxPolygon, xoff=inverseEgoTranslate[0],
                                                yoff=inverseEgoTranslate[1])
                        bboxPolygon = rotate(bboxPolygon, angle=inverseEgoAngle, origin=(0, 0))
                        currentDynamicObstaclesPolyOrin.append(bboxPolygon)
                        bboxPolygon = bboxPolygon.buffer(distanceToDynOb, cap_style=3)
                        currentDynamicObstaclesPoly.append(bboxPolygon)
                        currentDynamicObstaclesUUID[len(currentDynamicObstaclesPoly) - 1] = obstacle["Id"]
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
        initialSpeed = self._getInitialSpeed(listOfTime)
        logging.info("Median time tick {}, average time tick {}.".format(medianTimeTick, averageTimeTick))
        logging.info("InitialSpeed {} m/s.".format(initialSpeed))

        fotHyperparameters = STATIC_FOT_HYPERPARAMETERS

        # dynamic hyperparameters depending on speed
        fotHyperparameters["DT"] = averageTimeTick
        fotHyperparameters["MAX_T"] = lookForwardTime + averageTimeTick + 0.00001
        fotHyperparameters["MIN_T"] = lookForwardTime + averageTimeTick
        fotHyperparameters["TARGET_SPEED"] = fotHyperparameters["MIN_T"] * fotHyperparameters[
            "MAX_ACCEL"] * 0.50 + initialSpeed
        fotHyperparameters["D_T_S"] = abs(fotHyperparameters["TARGET_SPEED"] - initialSpeed) * 0.5 / fotHyperparameters[
            "N_S_SAMPLE"]
        print(fotHyperparameters["DT"], fotHyperparameters["MIN_T"], fotHyperparameters["TARGET_SPEED"],
              fotHyperparameters["D_T_S"], fotHyperparameters["D_T_S"] * fotHyperparameters["N_S_SAMPLE"])
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
        terminalSpeed = (
                    fotHyperparameters["TARGET_SPEED"] + fotHyperparameters["D_T_S"] * fotHyperparameters["N_S_SAMPLE"])
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
        timeBasedObstacles = self._getTimeBasedObstacles(listOfTime, staticObstaclesPoly, None, None)
        timeBasedObstaclesOrin = self._getTimeBasedObstaclesOrin(listOfTime, staticObstaclesPolyOrin, None, None)
        # prepare arguments for parallel execution
        emptyArguments = list()
        for o in range(len(setOfGoals)):
            emptyArgument = dict()
            referenceWaypoints = setOfGoals[o]
            emptyArgument["fotInitialState"] = fotInitialState
            emptyArgument["referenceWaypoints"] = referenceWaypoints
            emptyArgument["timeBasedObstacles"] = timeBasedObstacles
            emptyArgument["timeBasedObstaclesOrin"] = timeBasedObstaclesOrin
            emptyArgument["fotHyperparameters"] = fotHyperparameters
            emptyArgument["visualizeRoute"] = True
            emptyArgument["origin"] = origin
            emptyArgument["visualizeSaveName"] = "{}_golden_{}".format(timestamp, o)
            emptyArgument["removedList"] = []
            emptyArguments.append(emptyArgument)
        with ThreadPoolExecutor(max_workers=self.concurrentObjCount) as executor:
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
        timeBasedObstaclesOrin = self._getTimeBasedObstaclesOrin(listOfTime, staticObstaclesPolyOrin,
                                                                 dynamicObstaclesPolyOrin, dynamicObstaclesUUID)
        # prepare arguments for parallel execution
        fullArguments = list()
        for o in range(len(setOfGoalsAllObstacles)):
            fullArgument = dict()
            referenceWaypoints = setOfGoalsAllObstacles[o]
            fullArgument["fotInitialState"] = fotInitialState
            fullArgument["referenceWaypoints"] = referenceWaypoints
            fullArgument["timeBasedObstacles"] = timeBasedObstacles
            fullArgument["timeBasedObstaclesOrin"] = timeBasedObstaclesOrin
            fullArgument["fotHyperparameters"] = fotHyperparameters
            fullArgument["visualizeRoute"] = True
            fullArgument["origin"] = origin
            fullArgument["visualizeSaveName"] = "{}_full_{}".format(timestamp, o)
            fullArgument["removedList"] = []
            fullArguments.append(fullArgument)
        with ThreadPoolExecutor(max_workers=self.concurrentObjCount) as executor:
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
            dynamicObstaclesLessOnePolyOrin = copy.deepcopy(dynamicObstaclesPolyOrin)
            uuid = uuidCurrentTime[k]
            print("UUID to be removed at time {} is {}.".format(timestamp, uuid))
            lessOneSuccess = 0
            setOfGoalsLessOneObstacles = setOfGoals
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
            lessOneArguments = list()
            for o in range(len(setOfGoalsLessOneObstacles)):
                lessOneArgument = dict()
                referenceWaypoints = setOfGoalsLessOneObstacles[o]
                lessOneArgument["fotInitialState"] = fotInitialState
                lessOneArgument["referenceWaypoints"] = referenceWaypoints
                lessOneArgument["timeBasedObstacles"] = timeBasedObstaclesLessOne
                lessOneArgument["timeBasedObstaclesOrin"] = timeBasedObstaclesLessOneOrin
                lessOneArgument["timeBasedObstaclesVis"] = timeBasedObstaclesOrin
                lessOneArgument["fotHyperparameters"] = fotHyperparameters
                lessOneArgument["visualizeRoute"] = True
                lessOneArgument["origin"] = origin
                lessOneArgument["visualizeSaveName"] = "{}_lessone_{}_{}".format(timestamp, k, o)
                lessOneArgument["removedList"] = [uuid]
                lessOneArguments.append(lessOneArgument)
            with ThreadPoolExecutor(max_workers=self.concurrentObjCount) as executor:
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
    def riskVisualizer(obstacles, riskFactorDict, location, savedir, timestamp, visibleRange=70):
        fig, ax = plt.subplots(figsize=(3.5, 1.75), dpi=200)
        for idx, obstacle in enumerate(obstacles):
            if str(idx) in riskFactorDict:
                riskColor = riskFactorDict[str(idx)] / (max(riskFactorDict.values()) + 1e-6)
                riskColor = max(riskColor, 0)
                riskColor = (riskColor, 0, 0)
                xs, ys = obstacle.exterior.xy
                plt.fill(xs, ys, alpha=1.0, fc=riskColor, ec=riskColor)
            else:
                xs, ys = obstacle.exterior.xy
                plt.fill(xs, ys, alpha=1.0, fc="#999999", ec="#999999")
        plt.plot(location[0], location[1], "vc", markersize=2)
        ax.set_xlim(location[0] - visibleRange, location[0] + visibleRange)
        ax.set_ylim(location[1] - visibleRange / 2, location[1] + visibleRange / 2)
        ax.axis("off")
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig("{}/{}.jpg".format(savedir, timestamp))
        plt.close()

    def routingVisualizer(self, obstacles, origin, goals, savedir, savename,
                          removed=[], routes=None, refRoutes=None, visibleRange=80):
        # visualize environment before and after planning
        fig, ax = plt.subplots(figsize=(3.5, 1.75), dpi=200)
        # if refRoutes is not None and len(refRoutes[0]) and len(refRoutes[1]):
        #     plt.plot(refRoutes[0], refRoutes[1], "--", c='#fc4e2a', markersize=0.5)
        if routes is not None and len(routes):
            plt.plot(routes[0], routes[1], ".", c="#238b45", markersize=2)
        for goal in goals:
            plt.plot(goal[-1][0], goal[-1][1], "o", c="#bd0026", markersize=2)
        for t_obs in obstacles["static"]:
            for idx, obstacle in enumerate(t_obs):
                xs, ys = obstacle.exterior.xy
                plt.fill(xs, ys, alpha=1.0, fc='#999999', ec='#999999')
        for odx in range(len(obstacles["dynamic"])):
            for uuid, obstacle in obstacles["dynamic"][odx].items():
                xs, ys = obstacle.exterior.xy
                if uuid not in removed:
                    plt.fill(xs, ys, alpha=((odx + 1) / len(obstacles["dynamic"])), fc='#ffa114', ec='#ffa114')
                else:
                    plt.fill(xs, ys, alpha=((odx + 1) / len(obstacles["dynamic"])), fc='#1d91c0', ec='#1d91c0')
        plt.plot(origin[0], origin[1], "vc", markersize=2)
        ax.set_xlim(origin[0] - visibleRange, origin[0] + visibleRange)
        ax.set_ylim(origin[1] - visibleRange / 2, origin[1] + visibleRange / 2)
        ax.axis("off")
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig("{}/{}.jpg".format(savedir, savename))
        plt.close()


if __name__ == "__main__":
    basePath = path_configs.BASEPATH
    subFolder = sys.argv[1]
    logID = sys.argv[2]
    plannerType = sys.argv[3]
    assert plannerType == "fot*", "Planner type for trajectory based risk analysis can only be fot*"
    suffix = sys.argv[4]
    posUnc = sys.argv[5]
    timestamp = int(sys.argv[6])
    timestampIdx = int(sys.argv[7])
    concurrentObjCount = int(sys.argv[8])
    prediction = sys.argv[9]
    prefix = None
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
    trajDataPath = os.path.join(basePath, subFolder)
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
    if len(sys.argv) == 10:
        genRisk = GenerateRiskCarla(trajDataPath=trajDataPath,
                                    logID=logID,
                                    plannerType=plannerType,
                                    routeVisDir=routeVisDir,
                                    riskVisDir=riskVisDir,
                                    riskSaveDir=riskSaveDir,
                                    visualize=True,
                                    posUnc=posUnc,
                                    concurrentObjCount=concurrentObjCount,
                                    suffix=suffix,
                                    prediction=prediction)
    elif len(sys.argv) > 10:
        seed = sys.argv[10]
        genRisk = GenerateRiskCarla(trajDataPath=trajDataPath,
                                    logID=logID,
                                    plannerType=plannerType,
                                    routeVisDir=routeVisDir,
                                    riskVisDir=riskVisDir,
                                    riskSaveDir=riskSaveDir,
                                    plannerSeed=int(seed),
                                    visualize=True,
                                    posUnc=posUnc,
                                    concurrentObjCount=concurrentObjCount,
                                    suffix=suffix,
                                    prediction=prediction)
    else:
        raise ValueError
    genRisk.riskAnalyserSingleTimestamp(saveFileName="{}_risk_analysis.pkl.{}".format(suffix, timestamp), 
                                        timestamp=timestamp,
                                        timestampIdx=timestampIdx)
