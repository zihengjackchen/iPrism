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

import copy
import os
import time
from bev_planning_sim.generate_risk_traj_poly_single_timestamp_simfunction import GenerateRiskCarlaSimRuntime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle as pkl
import logging
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

from bev_planning_sim.frenet_hyperparameters import STATIC_FOT_HYPERPARAMETERS
from shapely import geometry
from shapely.affinity import translate, rotate
from bev_planning_sim.motion_prediction import *
from reachml.model.inference import ReachInferencer

logging.basicConfig(level=logging.ERROR)
# mpl.use('agg')
# print(mpl.get_backend())


class GenerateRiskCarlaSimRuntimeNN(GenerateRiskCarlaSimRuntime):
    def __init__(self, plannerType, suffix, routeVisDir, riskVisDir, riskSaveDir, 
                 dataSaveDir, dataVisDir, nn_model_path, nn_config_path,
                 concurrentObjCount, plannerSeed=0, visualize=False, posUnc="None", prediction='CVCTR'):
        super().__init__(plannerType, suffix, routeVisDir, riskVisDir, riskSaveDir,
                         concurrentObjCount, plannerSeed, visualize, posUnc, prediction)
        self.dataSaveDir = dataSaveDir
        self.visRange = 70
        self.vVisRange = 70 / 4
        self.W = 512
        self.H = 128
        self.DPI = 100
        self.pixelPerGridH = 11  # 3.5m roughly
        self.pixelPerGridW = 20  # 4.7m roughly
        self.reachableLabelsXDim = int(np.round(self.H / self.pixelPerGridH))
        self.reachableLabelsYDim = int(np.round(self.W / (2 * self.pixelPerGridW)))
        self.dataVisDir = dataVisDir
        print("Label dimension:", self.reachableLabelsXDim, self.reachableLabelsYDim)
        self.inferencer = ReachInferencer(nn_model_path, nn_config_path)

    def riskAnalyserSingleTimestamp(self, timestamp, timestampIdx, saveFileName=None,
                                    lookForwardDist=120, lookForwardTime=3):
        bookKeepingPerScene = dict()
        if timestamp in self.listofTimestamps:
            raArguement = dict()
            raArguement["timestamp"] = timestamp
            raArguement["lookForwardDist"] = lookForwardDist
            raArguement["lookForwardTime"] = lookForwardTime
            raArguement["timestampIdx"] = timestampIdx
            result = self._riskAnalyserOneTimestampNN(raArguement)
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
        return bookKeepingPerScene

    def _collectFeatureMapsSlow(self, timeBasedObstacles, savedir, savename, origin=(0, 0)):
        plt.style.use('dark_background')
        featureMaps = list()
        W = self.W
        H = self.H
        DPI = self.DPI
        for odx in range(len(timeBasedObstacles["dynamic"])):
            fig, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)
            for laneId in self.rasterizedOptimizedLanesAroundTraj['uniqueLaneIDs']:
                for laneSurface in self.rasterizedOptimizedLanesAroundTraj['processedLane'][laneId]['rasterizedSurfaceTrans']:
                    xs, ys = laneSurface.exterior.xy
                    plt.fill(xs, ys, alpha=1.0, fc='white', ec='white')
                for idx, obstacle in enumerate(timeBasedObstacles["static"][odx]):
                    xs, ys = obstacle.exterior.xy
                    plt.fill(xs, ys, alpha=1.0, fc='black', ec='black')
                for uuid, obstacle in timeBasedObstacles["dynamic"][odx].items():
                    xs, ys = obstacle.exterior.xy
                    plt.fill(xs, ys, alpha=1.0, fc='black', ec='black')
            ax.set_xlim(origin[0] - self.visRange, origin[0] + self.visRange)
            ax.set_ylim(origin[1] - self.vVisRange, origin[1] + self.vVisRange)
            ax.axis("off")
            plt.subplots_adjust(left=0.00, right=1.0, top=1.0, bottom=0.00)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
            data /= 255
            featureMaps.append(data)
            # plt.savefig("{}/{}.pdf".format(savedir, savename))
            plt.close()
            # plt.figure()
            # plt.imshow(data)
            # plt.show()
        featureMaps = np.array(featureMaps)
        plt.style.use('default')
        return featureMaps

    def _collectFeatureMaps(self, timeBasedObstacles, savedir, savename, origin=(0, 0), base_img=None):
        featureMaps = list()
        perPixelDistanceX = self.visRange / (self.W / 2 - 1)
        perPixelDistanceY = self.vVisRange / (self.H / 2 - 1)
        if base_img is None:
            base_img = np.zeros((self.H, self.W))
            for laneId in self.rasterizedOptimizedLanesAroundTraj['uniqueLaneIDs']:
                for laneSurface in self.rasterizedOptimizedLanesAroundTraj['processedLane'][laneId]['rasterizedSurfaceTrans']:
                    xs, ys = laneSurface.exterior.xy
                    xs, ys = np.array(xs), np.array(ys)
                    pixelXs = np.round((xs + self.visRange) / perPixelDistanceX).astype(int)
                    pixelYs = np.round((-ys + self.vVisRange) / perPixelDistanceY).astype(int)
                    localPoints = np.stack((pixelXs,pixelYs), axis=1)
                    localPoints = np.unique(localPoints, axis=0)
                    delaunaryHull = Delaunay(localPoints)
                    localPointXs, localPointYs = localPoints[delaunaryHull.convex_hull,0], localPoints[delaunaryHull.convex_hull,1]
                    maxX = int(np.max(localPointXs))
                    maxY = int(np.max(localPointYs))
                    minX = int(np.min(localPointXs))
                    minY = int(np.min(localPointYs))
                    for pixelX in range(minX, maxX + 1, 1):
                        for pixelY in range(minY, maxY + 1, 1):
                            if 0 <= pixelX < base_img.shape[1] and 0 <= pixelY < base_img.shape[0]:
                                pixelPoint = [pixelX, pixelY]
                                if delaunaryHull.find_simplex(pixelPoint) >= 0:
                                    base_img[pixelY, pixelX] = 1
        for odx in range(len(timeBasedObstacles["dynamic"])):
            stack_img = copy.deepcopy(base_img)
            for _, obstacle in timeBasedObstacles["dynamic"][odx].items():
                xs, ys = obstacle.exterior.xy
                xs, ys = np.array(xs), np.array(ys)
                pixelXs = np.round((xs + self.visRange) / perPixelDistanceX).astype(int)
                pixelYs = np.round((-ys + self.vVisRange) / perPixelDistanceY).astype(int)
                localPoints = np.stack((pixelXs,pixelYs), axis=1)
                localPoints = np.unique(localPoints, axis=0)
                delaunaryHull = Delaunay(localPoints)
                localPointXs, localPointYs = localPoints[delaunaryHull.convex_hull,0], localPoints[delaunaryHull.convex_hull,1]
                maxX = int(np.max(localPointXs))
                maxY = int(np.max(localPointYs))
                minX = int(np.min(localPointXs))
                minY = int(np.min(localPointYs))
                for pixelX in range(minX, maxX + 1, 1):
                    for pixelY in range(minY, maxY + 1, 1):
                        if 0 <= pixelX < stack_img.shape[1] and 0 <= pixelY < stack_img.shape[0]:
                            pixelPoint = [pixelX, pixelY]
                            if delaunaryHull.find_simplex(pixelPoint) >= 0:
                                stack_img[pixelY, pixelX] = 0
            featureMaps.append(stack_img)
        featureMaps = np.array(featureMaps)
        return featureMaps, base_img

    def _collectReachableResults(self, setOfGoals, predictedLabelsFlat, featureMaps):
        # first set all goals to be not reachable
        successGoals = list()
        predictedLabelsFlat = predictedLabelsFlat.reshape((1, self.reachableLabelsXDim, self.reachableLabelsYDim))
        perPixelDistanceX = self.visRange / (self.W / 2 - 1)
        perPixelDistanceY = self.vVisRange / (self.H / 2 - 1)

        for goal in setOfGoals:
            goal = goal[-1]
            goalX = goal[0]
            goalY = goal[1]
            # first convert to pixel location
            goalPixelX = np.round(goalX / perPixelDistanceX)
            goalPixelY = np.round((-goalY + self.vVisRange) / perPixelDistanceY)

            # DEBUG: plot the pixel location
            # for featureMap in featureMaps:
            #     featureMap[goalPixelY, goalPixelX] = 0
            #     plt.imshow(featureMap)
            #     plt.show()

            # then convert to grid coordinates top-left
            gridY = int(goalPixelX // self.pixelPerGridW)
            gridX = int(goalPixelY // self.pixelPerGridH)

            # DEBUG: plot the grid location
            # featureMap = copy.deepcopy(featureMaps[0])
            # featureMap[gridX * self.pixelPerGridH:(gridX + 1) * self.pixelPerGridH, gridY * self.pixelPerGridW:(gridY + 1) * self.pixelPerGridW] = 0
            # plt.imshow(featureMap)
            # plt.show()

            # finally, set one-hot labels, change the goals to reachable
            if gridY < predictedLabelsFlat.shape[2] and gridX < predictedLabelsFlat.shape[1]:
                if predictedLabelsFlat[:, gridX, gridY] >=0.5:
                    successGoals.append(goal)
        return successGoals

    @torch.no_grad()
    def _nn_inference(self, bev_features, ego_states):
        bev_features = bev_features[None, :]
        ego_states = ego_states[None, :]
        predicted_labels = self.inferencer.inference_wrapper(bev_features, ego_states)
        predicted_labels_flat = torch.flatten(predicted_labels.cpu()).detach().numpy()
        return predicted_labels_flat

    def _riskAnalyserOneTimestampNN(self, raArguement):
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
            return [], [], []

        # taka list of dynamic obstacles that aligns with the time stamps
        dynamicObstacles = list()

        # choose to use ground truth data
        if self.prediction == "GT":
            raise RuntimeError("For simulation cannot use GT.")
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
        # plt.show()
        # raise NotImplementedError

        # 1. find the next set of POI based on the current location
        setOfGoals = []
        for laneID in self.rasterizedOptimizedLanesAroundTraj["uniqueLaneIDs"]:
            lanePOI = copy.deepcopy(self.rasterizedOptimizedLanesAroundTraj["processedLane"][laneID]["lanePOI"])[:, 0:2]
            lanePOIOrin = copy.deepcopy(lanePOI)
            lanePOIOrinPrev = copy.deepcopy(
                self.rasterizedOptimizedLanesAroundTraj["processedLane"][laneID]["lanePOIPrev"])[:, 0:2]
            if len(lanePOI):
                lanePOI[:, 0] = lanePOI[:, 0] + inverseEgoTranslate[0]
                lanePOI[:, 1] = lanePOI[:, 1] + inverseEgoTranslate[1]
                inverseRotationMatrix = np.array([
                    [np.cos(np.deg2rad(inverseEgoAngle)), -np.sin(np.deg2rad(inverseEgoAngle))],
                    [np.sin(np.deg2rad(inverseEgoAngle)), np.cos(np.deg2rad(inverseEgoAngle))]
                ])
                lanePOI = lanePOI @ inverseRotationMatrix.T
                for idx, POI in enumerate(lanePOI):
                    # we use a simplified version for sim inference, since NN is used, no inference path is needed
                    if minDist < POI[0] <= lookForwardDist:
                        POI = [lanePOIOrin[idx][0:2]]
                        POI = np.array(POI)
                        POI[:, 0] = POI[:, 0] + inverseEgoTranslate[0]
                        POI[:, 1] = POI[:, 1] + inverseEgoTranslate[1]
                        POI = POI @ inverseRotationMatrix.T
                        setOfGoals.append(list(POI))

        # 2. origin as current position, list of static boundary obstacles, and goal as POI
        setOfGoals = sorted(setOfGoals, key=lambda element: (element[0][0]))
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
            static_obs_bbox_poly = translate(static_obs_bbox_poly, xoff=inverseEgoTranslate[0],
                                             yoff=inverseEgoTranslate[1])
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
        dynamicObstaclesPoly = list()
        dynamicObstaclesPolyOrin = list()
        dynamicObstaclesUUID = list()

        # the risk is still single timestamp
        riskDict = dict()
        riskFactorRawDict = dict()
        riskFactorDict = dict()
        featureMaps = list()
        reachableLabels = list()
        egoStates = list()

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

        # generate hyperparameters for the frenet planner
        """
        1. hyperparameters that are frame and parameter independent: max speed, max acc, max cur, max road width,
           road width sample delta, targeted speed sample delta, sample number of speed, distance buffer to obstacle,
           threshold distance to goal, and all lost cost parameters.
        2. hyperparameters that are frame dependent:  time tick, and targeted speed (the end speed)
        """

        averageTimeTick = self._getAverageTimeTick(listOfTime)
        medianTimeTick = self._getMedianTimeTick(listOfTime)
        initialSpeed = self._getInitialSpeed(timestampIdx, timestampIdx-1)

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
        # print("Before purging:", len(setOfGoals))
        reachableGoal = []
        terminalSpeed = (
                fotHyperparameters["TARGET_SPEED"] + fotHyperparameters["D_T_S"] * fotHyperparameters["N_S_SAMPLE"])
        theoryReachable = ((terminalSpeed + initialSpeed) / 2) * fotHyperparameters["MIN_T"]
        for waypoints in setOfGoals:
            dest = waypoints[-1]
            if np.linalg.norm(dest - origin) < theoryReachable:
                reachableGoal.append(waypoints)
        setOfGoals = reachableGoal
        # print("After purging:", len(setOfGoals))

        # 4. check if the goal is reachable or not with removing one obstacle per time
        emptySuccess = 0
        timeBasedObstacles = self._getTimeBasedObstacles(listOfTime, staticObstaclesPoly, None, None)
        # timeBasedObstaclesOrin = self._getTimeBasedObstaclesOrin(listOfTime, staticObstaclesPolyOrin, None, None)

        # prepare arguments for nn inference
        successGoals = list()
        emptyObstacleFeatureMaps, baseImg = self._collectFeatureMaps(timeBasedObstacles, "", "")
        egoState = np.array([fotInitialState["c_speed"],
                            fotInitialState["c_d"],
                            fotInitialState["c_d_d"],
                            fotInitialState["c_d_dd"],
                            fotInitialState["s0"]])
        predicted_labels_flat = self._nn_inference(emptyObstacleFeatureMaps, egoState)
        successResults = self._collectReachableResults(setOfGoals, predicted_labels_flat, emptyObstacleFeatureMaps)
        for successGoal in successResults:
            emptySuccess += 1
            successGoals.append(successGoal)
        if not emptySuccess:
            logging.warning("Warning there is no success but we will continue anyway.")
        # print("Done evaluating empty grid, empty success:", emptySuccess)
        # featureMaps.append(emptyObstacleFeatureMaps)
        # egoStates.append(np.array([fotInitialState["c_speed"],
        #             fotInitialState["c_d"],
        #             fotInitialState["c_d_d"],
        #             fotInitialState["c_d_dd"],
        #             fotInitialState["s0"]]))


        fullSuccess = 0
        setOfGoalsAllObstacles = setOfGoals
        timeBasedObstacles = self._getTimeBasedObstacles(listOfTime, staticObstaclesPoly,
                                                         dynamicObstaclesPoly, dynamicObstaclesUUID)
        # timeBasedObstaclesOrin = self._getTimeBasedObstaclesOrin(listOfTime, staticObstaclesPolyOrin,
        #                                                          dynamicObstaclesPolyOrin, dynamicObstaclesUUID)
        # prepare arguments for nn inference
        successGoals = list()
        fullObstaclesFeatureMaps, baseImg = self._collectFeatureMaps(timeBasedObstacles, "", "", base_img=baseImg)
        egoState = np.array([fotInitialState["c_speed"],
                            fotInitialState["c_d"],
                            fotInitialState["c_d_d"],
                            fotInitialState["c_d_dd"],
                            fotInitialState["s0"]])
        predicted_labels_flat = self._nn_inference(fullObstaclesFeatureMaps, egoState)
        successResults = self._collectReachableResults(setOfGoalsAllObstacles, predicted_labels_flat, fullObstaclesFeatureMaps)
        for successGoal in successResults:
            fullSuccess += 1
            successGoals.append(successGoal)
        # print("Done evaluating fully occupied grid, full obstacle success:", fullSuccess)
        # featureMaps.append(fullObstaclesFeatureMaps)
        # egoStates.append(np.array([fotInitialState["c_speed"],
        #                            fotInitialState["c_d"],
        #                            fotInitialState["c_d_d"],
        #                            fotInitialState["c_d_dd"],
        #                            fotInitialState["s0"]]))

        # 5. Finally remove obstacles one by one
        uuidCurrentTime = list(dynamicObstaclesUUID[0].values())
        for k in range(len(uuidCurrentTime)):
            dynamicObstaclesLessOnePoly = copy.deepcopy(dynamicObstaclesPoly)
            # dynamicObstaclesLessOnePolyOrin = copy.deepcopy(dynamicObstaclesPolyOrin)
            uuid = uuidCurrentTime[k]
            # print("UUID to be removed at time {} is {}.".format(timestamp, uuid))
            lessOneSuccess = 0
            successGoals = list()
            setOfGoalsLessOneObstacles = setOfGoals
            timeBasedObstaclesLessOne = self._getTimeBasedObstacles(listOfTime,
                                                                    staticObstaclesPoly,
                                                                    dynamicObstaclesLessOnePoly,
                                                                    dynamicObstaclesUUID,
                                                                    [uuid])
            # timeBasedObstaclesLessOneOrin = self._getTimeBasedObstaclesOrin(listOfTime,
            #                                                                 staticObstaclesPolyOrin,
            #                                                                 dynamicObstaclesLessOnePolyOrin,
            #                                                                 dynamicObstaclesUUID,
            #                                                                 [uuid])
            # prepare arguments for parallel execution
            lessOneObstaclesFeatureMaps, baseImg = self._collectFeatureMaps(timeBasedObstaclesLessOne, "", "", base_img=baseImg)
            egoState = np.array([fotInitialState["c_speed"],
                                fotInitialState["c_d"],
                                fotInitialState["c_d_d"],
                                fotInitialState["c_d_dd"],
                                fotInitialState["s0"]])
            predicted_labels_flat = self._nn_inference(lessOneObstaclesFeatureMaps, egoState)
            successResults = self._collectReachableResults(setOfGoalsLessOneObstacles, predicted_labels_flat, lessOneObstaclesFeatureMaps)
            for successGoal in successResults:
                lessOneSuccess += 1
                successGoals.append(successGoal)
            if emptySuccess:
                riskFactor = (lessOneSuccess - fullSuccess) * 1.0 / emptySuccess
            else:
                riskFactor = 0.0
            riskDict[str(k)] = lessOneSuccess
            riskFactorRawDict[str(k)] = riskFactor
            riskFactor = max(riskFactor, 0)
            riskFactorDict[str(k)] = riskFactor
            del dynamicObstaclesLessOnePoly

            # !! DC: Collect less than one feature maps
            # if len(uuidCurrentTime) > 1:
            #     featureMaps.append(lessOneObstaclesFeatureMaps)
            #     egoStates.append(np.array([fotInitialState["c_speed"],
            #                                fotInitialState["c_d"],
            #                                fotInitialState["c_d_d"],
            #                                fotInitialState["c_d_dd"],
            #                                fotInitialState["s0"]]))
        riskDict["empty"] = emptySuccess
        riskDict["full"] = fullSuccess
        # print("Done analysing one frame after {} seconds.".format(time.time() - startTime))
        print(riskFactorRawDict, riskFactorDict, riskDict, dynamicObstaclesUUID[0])
        if self.visualize:
            self.riskVisualizer(np.array(dynamicObstaclesPolyOrin[0] + staticObstaclesPolyOrin),
                                riskFactorDict, origin, self.riskVisDir, timestamp)
            self.dataVisualizer(featureMaps, reachableLabels, timestamp)
        return riskDict, riskFactorDict, riskFactorRawDict, dynamicObstaclesUUID, dynamicObstaclesPoly

    def dataVisualizer(self, featureMaps, reachableLabels, timestamp):
        for mIdx, featureMap in enumerate(featureMaps):
            currentTimeFeatureMap = featureMap[0]
            featureImgPath = os.path.join(self.dataVisDir, "featureMaps_{}_{}.png".format(timestamp, mIdx))
            plt.imsave(featureImgPath, currentTimeFeatureMap, vmin=0, vmax=1.0)

        for lIdx, labelMap in enumerate(reachableLabels):
            currentTimeFeatureMap = featureMaps[lIdx][0]
            baseImg = copy.deepcopy(currentTimeFeatureMap) * 0.5
            baseImg = baseImg[:, int(np.floor(baseImg.shape[1] / 2)):]
            labelMap = copy.deepcopy(labelMap)
            labelMap = np.squeeze(labelMap, axis=0)
            for r in range(labelMap.shape[0]):
                for c in range(labelMap.shape[1]):
                    if labelMap[r, c] == 1:
                        baseImg[r * self.pixelPerGridH:(r + 1) * self.pixelPerGridH,
                                c * self.pixelPerGridW:(c + 1) * self.pixelPerGridW] = labelMap[r, c] * 1
                    labelImgPath = os.path.join(self.dataVisDir, "labels_{}_{}.png".format(timestamp, lIdx))
                    plt.imsave(labelImgPath, baseImg, vmin=0, vmax=1.0)
