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
import gc
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import sys
import logging
import path_configs

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.frame_label_accumulator import PerFrameLabelAccumulator
from argoverse.utils.interpolate import interp_arc
from shapely import geometry
from shapely.affinity import translate, rotate
from scipy.spatial.transform import Rotation as R

IS_OCCLUDED_FLAG = 100
LANE_TANGENT_VECTOR_SCALING = 4

VIS = True


# Used in the GenerateRisk class for risk analysis with planning algorithms
class GenerateBEVGraph:
    def __init__(self, mapDataPath, trajDataPath, logID, experimentPrefix="default"):
        self.mapDataPath = mapDataPath
        self.trajDataPath = trajDataPath
        self.argoverseMap = ArgoverseMap(self.mapDataPath)
        self.argoverseTrajs = ArgoverseTrackingLoader(self.trajDataPath)
        self.listofTraj = self.argoverseTrajs.log_list
        self.currentTrajID = logID
        self.cityName = self.argoverseTrajs.get(self.currentTrajID).city_name
        assert self.currentTrajID in self.listofTraj
        # load current labels
        self.pfa = PerFrameLabelAccumulator(self.trajDataPath, self.trajDataPath, experimentPrefix, save=False)
        self.pfa.accumulate_per_log_data(log_id=self.currentTrajID)
        self.perCityTrajDict = self.pfa.per_city_traj_dict
        self.logEgoPoseDict = self.pfa.log_egopose_dict
        self.logTimestampDict = self.pfa.log_timestamp_dict
        self.listofTimestamps = self._parseLidarTimestamps()
        self.optimizedLanesAroundTraj = None
        self.rasterizedOptimizedLanesAroundTraj = None
        print("Done parsing driving trajectory log {}.".format(self.currentTrajID))

    def _parseLidarTimestamps(self):
        lidarLogPath = os.path.join(self.trajDataPath, self.currentTrajID, 'lidar')
        lidarLogFiles = sorted([f for f in os.listdir(lidarLogPath) if os.path.isfile(os.path.join(lidarLogPath, f))])
        lidarLogTimestamp = list()
        for f in lidarLogFiles:
            f = str(f)
            parsedTimestamp = (f.split('_')[-1]).split('.')[0]
            parsedTimestamp = int(parsedTimestamp)
            lidarLogTimestamp.append(parsedTimestamp)
        return lidarLogTimestamp

    @staticmethod
    def _angleBetweenVectors(vector1, vector2):
        unitVector1 = vector1 / np.linalg.norm(vector1)
        unitVector2 = vector2 / np.linalg.norm(vector2)
        dotProduct = np.dot(unitVector1, unitVector2)
        angle = np.arccos(dotProduct)
        return angle

    def _getLanesAroundCurrentLoc(self, timestamp, centerLineSampling=50):
        logging.info("Get same direction lanes around current location at timestamp: {}".format(timestamp))
        lanesInfoDict = {}
        laneIDs = []
        laneCenterLines = []
        lanePolygons = []
        poseCitytoEgo = self.logEgoPoseDict[self.currentTrajID][timestamp]
        egoTranslation = np.array(poseCitytoEgo["translation"])
        closestLane, conf, centerLine = self.argoverseMap.get_nearest_centerline(egoTranslation[0:2],
                                                                                 self.cityName, False)
        laneID = closestLane.id
        laneIDs.append([laneID, "ego"])
        # self.argoverseMap.draw_lane(laneID, self.cityName)
        # plt.show()
        centerLine = self.argoverseMap.get_lane_segment_centerline(laneID, self.cityName)
        centerLineDense = interp_arc(centerLineSampling, centerLine[:, 0], centerLine[:, 1])
        laneCenterLines.append([centerLineDense, "ego"])

        # helper function check if lanes traveling in the same direction
        def _sameDirection(lane1, lane2):
            laneCenterLine1 = self.argoverseMap.get_lane_segment_centerline(lane1, self.cityName)
            laneCenterLine2 = self.argoverseMap.get_lane_segment_centerline(lane2, self.cityName)
            coLinear = True
            for idx in range(min(len(laneCenterLine1), len(laneCenterLine2)) - 1):
                lane1Direction = laneCenterLine1[idx + 1] - laneCenterLine1[idx]
                lane2Direction = laneCenterLine2[idx + 1] - laneCenterLine2[idx]
                angle = self._angleBetweenVectors(lane1Direction, lane2Direction)
                if angle >= np.pi / 18:
                    coLinear = False
            return coLinear

        # search all neighboring lanes to the left
        currentID = laneID
        currentLane = copy.deepcopy(closestLane)
        leftCount = 0
        while currentID:
            leftLaneID = currentLane.l_neighbor_id
            if leftLaneID and _sameDirection(currentID, leftLaneID):
                laneIDs.append([leftLaneID, "left_{}".format(leftCount)])
                centerLine = self.argoverseMap.get_lane_segment_centerline(leftLaneID, self.cityName)
                centerLineDense = interp_arc(centerLineSampling, centerLine[:, 0], centerLine[:, 1])
                laneCenterLines.append([centerLineDense, "left_{}".format(leftCount)])
                leftCount += 1
            else:
                break
            currentID = leftLaneID
            currentLane = copy.deepcopy(self.argoverseMap.city_lane_centerlines_dict[self.cityName][leftLaneID])
        # search all neighboring lanes to the right
        currentID = laneID
        currentLane = copy.deepcopy(closestLane)
        rightCount = 0
        while currentID:
            rightLaneID = currentLane.r_neighbor_id
            if rightLaneID and _sameDirection(currentID, rightLaneID):
                laneIDs.append([rightLaneID, "right_{}".format(rightCount)])
                centerLine = self.argoverseMap.get_lane_segment_centerline(rightLaneID, self.cityName)
                centerLineDense = interp_arc(centerLineSampling, centerLine[:, 0], centerLine[:, 1])
                laneCenterLines.append([centerLineDense, "right_{}".format(rightCount)])
                rightCount += 1
            else:
                break
            currentID = rightLaneID
            currentLane = copy.deepcopy(self.argoverseMap.city_lane_centerlines_dict[self.cityName][rightLaneID])
        lanesInfoDict["laneIDs"] = laneIDs
        for laneID in laneIDs:
            lanePolygons.append(self.argoverseMap.get_lane_segment_polygon(laneID[0], self.cityName))
        lanesInfoDict["lanePolygons"] = lanePolygons
        lanesInfoDict["laneCenterLines"] = laneCenterLines
        return lanesInfoDict

    def _lanestoKeepIntersect(self, lanesAroundTraj):
        lanesIntersection = {}
        egoInterTranslations = []
        allLanestoKeep = {}
        startedIntersection = False
        startTimeStamp = 0

        # inner function that checks which lane in the intersection is the candidate lane based on
        # the vehicle trajectory
        def _getLanestoKeep(egoInterTranslations, lanesIntersection, sampleNum=50):
            egoInterTranslations = np.array(egoInterTranslations)
            egoInterDenseTraj = interp_arc(sampleNum, egoInterTranslations[:, 0], egoInterTranslations[:, 1])
            accuErrors = []
            toKeep = []
            for laneID, laneCenterLine in lanesIntersection.items():
                toKeep.append(laneID)
                laneDenseCenterLine = interp_arc(sampleNum, laneCenterLine[:, 0], laneCenterLine[:, 1])
                angles = []
                for idx in range(min(len(laneDenseCenterLine), len(egoInterDenseTraj)) - 1):
                    laneVector = laneDenseCenterLine[idx + 1] - laneDenseCenterLine[idx]
                    egoVector = egoInterDenseTraj[idx + 1] - egoInterDenseTraj[idx]
                    angle = self._angleBetweenVectors(laneVector, egoVector)
                    angles.append(angle)
                accuErrors.append(np.sum(angles))
            minError = np.argmin(accuErrors)
            return toKeep[int(minError)]

        for idx, timestamp in enumerate(self.listofTimestamps):
            laneIDs = lanesAroundTraj[timestamp]["laneIDs"]
            for laneID in laneIDs:
                laneSegment = self.argoverseMap.city_lane_centerlines_dict[self.cityName][laneID[0]]
                # not yet at intersection but now we just entered one
                if laneSegment.is_intersection and laneID[1] == "ego" and len(egoInterTranslations) == 0:
                    lanesIntersection[laneID[0]] = self.argoverseMap.city_lane_centerlines_dict[self.cityName][
                        laneID[0]].centerline
                    poseCitytoEgo = self.logEgoPoseDict[self.currentTrajID][timestamp]
                    egoTranslation = np.array(poseCitytoEgo["translation"])
                    egoInterTranslations.append(egoTranslation)
                    startedIntersection = True
                    startTimeStamp = timestamp
                elif laneSegment.is_intersection and laneID[1] == "ego" and startedIntersection:
                    lanesIntersection[laneID[0]] = self.argoverseMap.city_lane_centerlines_dict[self.cityName][
                        laneID[0]].centerline
                    poseCitytoEgo = self.logEgoPoseDict[self.currentTrajID][timestamp]
                    egoTranslation = np.array(poseCitytoEgo["translation"])
                    egoInterTranslations.append(egoTranslation)
                # we have exited the intersection at this point
                elif not laneSegment.is_intersection and laneID[1] == "ego" and len(egoInterTranslations) > 0:
                    lanestoKeep = _getLanestoKeep(egoInterTranslations, lanesIntersection)
                    allLanestoKeep[(startTimeStamp, self.listofTimestamps[idx - 1])] = lanestoKeep
                    # resetting for the next intersection
                    lanesIntersection = {}
                    egoInterTranslations = []
                    startedIntersection = False
                    startTimeStamp = 0
                else:
                    continue
        return allLanestoKeep

    def _postProcessLanes(self, lanesAroundTraj, allLanestoKeepAtIntersect):
        retLanesAroundTrajLite = {}
        uniqueLanes = []

        def _atIntersect(timestamp, allLanestoKeepAtIntersect):
            for timeTuple in allLanestoKeepAtIntersect:
                if timeTuple[0] <= timestamp <= timeTuple[1]:
                    return timeTuple
            return None

        for timestamp in self.listofTimestamps:
            laneIDs = lanesAroundTraj[timestamp]["laneIDs"]
            timetoLanesDict = {"lanes": [], "leftMostLaneID": -1, "rightMostLaneID": -1, "egoLaneID": -1}
            leftMaxID = -1
            leftMaxLane = None
            rightMaxID = -1
            rightMaxLane = None
            for laneID in laneIDs:
                timeAtIntersect = _atIntersect(timestamp, allLanestoKeepAtIntersect)
                if timeAtIntersect is not None and laneID[0] != allLanestoKeepAtIntersect[timeAtIntersect]:
                    logging.debug("timestamp: {} at intersection, replace laneID: {} with {}".format(timestamp, laneID[0],
                                                                                                     allLanestoKeepAtIntersect[
                                                                                                     timeAtIntersect]))
                    timetoLanesDict["egoLaneID"] = allLanestoKeepAtIntersect[timeAtIntersect]
                    timetoLanesDict["lanes"].append([allLanestoKeepAtIntersect[timeAtIntersect], "ego"])
                else:
                    if "ego" in laneID:
                        timetoLanesDict["egoLaneID"] = laneID[0]
                    if "left" in laneID[1]:
                        idx = int(laneID[1].split("_")[-1])
                        if idx > leftMaxID:
                            leftMaxID = idx
                            leftMaxLane = laneID[0]
                    if "right" in laneID[1]:
                        idx = int(laneID[1].split("_")[-1])
                        if idx > rightMaxID:
                            rightMaxID = idx
                            rightMaxLane = laneID[0]
                    timetoLanesDict["lanes"].append(laneID)
            if leftMaxID >= 0:
                timetoLanesDict["leftMostLaneID"] = leftMaxLane
            if rightMaxID >= 0:
                timetoLanesDict["rightMostLaneID"] = rightMaxLane
            retLanesAroundTrajLite[timestamp] = copy.deepcopy(timetoLanesDict)

        for timestamp in self.listofTimestamps:
            uniqueLanes.extend([item[0] for item in retLanesAroundTrajLite[timestamp]["lanes"]])
        retLanesAroundTrajLite["uniqueLaneIDs"] = set(uniqueLanes)
        return retLanesAroundTrajLite

    def getLanesAroundCurrentTraj(self):
        lanesAroundTraj = {}
        for timestamp in self.listofTimestamps:
            lanesInfoDict = self._getLanesAroundCurrentLoc(timestamp)
            lanesAroundTraj[timestamp] = copy.deepcopy(lanesInfoDict)
        allLanestoKeepAtIntersect = self._lanestoKeepIntersect(lanesAroundTraj)
        optimizedLanesAroundTraj = self._postProcessLanes(lanesAroundTraj, allLanestoKeepAtIntersect)
        self.optimizedLanesAroundTraj = copy.deepcopy(optimizedLanesAroundTraj)
        return optimizedLanesAroundTraj

    def _calculateOutReachfromNormal(self, denseCenterLine, idx, laneWidth):
        if idx == 0:
            centerPoints = np.array([denseCenterLine[idx], denseCenterLine[idx + 1]])
        else:
            centerPoints = np.array([denseCenterLine[idx - 1], denseCenterLine[idx]])
        denseLaneVec = interp_arc(10, centerPoints[:, 0], centerPoints[:, 1])
        avgDx = np.average(np.gradient(denseLaneVec[:, 0]))
        avgDy = np.average(np.gradient(denseLaneVec[:, 1]))

        # handle special case with vertical slopes
        if avgDx == 0:
            print("Encounter special case in the outreach function, vertical motion.")
            # travelling downwards
            if denseLaneVec[0, 1] > denseLaneVec[-1, 1]:
                leftSide = [denseCenterLine[idx][0] + laneWidth / 2.0,  denseCenterLine[idx][1]]
                rightSide = [denseCenterLine[idx][0] - laneWidth / 2.0, denseCenterLine[idx][1]]
                leftSideBound = [denseCenterLine[idx][0] + laneWidth / 2.0 + 0.1, denseCenterLine[idx][1]]
                rightSideBound = [denseCenterLine[idx][0] - laneWidth / 2.0 - 0.1, denseCenterLine[idx][1]]
            # travelling upwards
            elif denseLaneVec[0, 1] < denseLaneVec[-1, 1]:
                leftSide = [denseCenterLine[idx][0] - laneWidth / 2.0, denseCenterLine[idx][1]]
                rightSide = [denseCenterLine[idx][0] + laneWidth / 2.0, denseCenterLine[idx][1]]
                leftSideBound = [denseCenterLine[idx][0] - laneWidth / 2.0 - 0.1, denseCenterLine[idx][1]]
                rightSideBound = [denseCenterLine[idx][0] + laneWidth / 2.0 + 0.1, denseCenterLine[idx][1]]
            else:
                raise Exception("The waypoints must move.")
            return leftSide, rightSide, leftSideBound, rightSideBound

        # calculate slope
        slope = avgDy / avgDx

        # handle special case with horizontal slopes
        if slope == 0:
            print("Encounter special case in the outreach function, horizontal motion.")
            # travelling left
            if denseLaneVec[0, 0] > denseLaneVec[-1, 0]:
                leftSide = [denseCenterLine[idx][0], denseCenterLine[idx][1] - laneWidth / 2.0]
                rightSide = [denseCenterLine[idx][0], denseCenterLine[idx][1] + laneWidth / 2.0]
                leftSideBound = [denseCenterLine[idx][0], denseCenterLine[idx][1] - laneWidth / 2.0 - 0.1]
                rightSideBound = [denseCenterLine[idx][0], denseCenterLine[idx][1] + laneWidth / 2.0 + 0.1]
            # travelling right
            elif denseLaneVec[0, 0] < denseLaneVec[-1, 0]:
                leftSide = [denseCenterLine[idx][0], denseCenterLine[idx][1] + laneWidth / 2.0]
                rightSide = [denseCenterLine[idx][0], denseCenterLine[idx][1] - laneWidth / 2.0]
                leftSideBound = [denseCenterLine[idx][0], denseCenterLine[idx][1] + laneWidth / 2.0 + 0.1]
                rightSideBound = [denseCenterLine[idx][0], denseCenterLine[idx][1] - laneWidth / 2.0 - 0.1]
            else:
                raise Exception("The waypoints must move.")
            return leftSide, rightSide, leftSideBound, rightSideBound

        invSlope = -1.0 / slope
        # in radiant
        theta = np.arctan(invSlope)
        xDiff = laneWidth / 2.0 * np.cos(theta)
        yDiff = laneWidth / 2.0 * np.sin(theta)
        xBoundDiff = 0.1 * np.cos(theta)
        yBoundDiff = 0.1 * np.sin(theta)
        leftSide = [denseCenterLine[idx][0] - xDiff, denseCenterLine[idx][1] - yDiff]
        rightSide = [denseCenterLine[idx][0] + xDiff, denseCenterLine[idx][1] + yDiff]
        leftSideBound = [denseCenterLine[idx][0] - xDiff - xBoundDiff, denseCenterLine[idx][1] - yDiff - yBoundDiff]
        rightSideBound = [denseCenterLine[idx][0] + xDiff + xBoundDiff, denseCenterLine[idx][1] + yDiff + yBoundDiff]
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

    def getRasterizedLanesDistTraj(self, optimizedLanesAroundTraj, travelDist=4.5,
                                   laneWidth=4.5, visualize=False):
        """
        This function takes in a set of lanes information along the current trajectory and constructs
        the rasterized version of the dense center line of a lane according to traveling distance and
        the width of the lane in meters. The approximate center of the rasterized grid is then a candidate
        way point that can be used as a destination point for the vehicle to travel to.
        The function returns the following things:
            1. The set of possible destinations along the center line of the lanes (unordered)
            2. The set of polygons of the rasterized lane surfaces (ignore z the elevation)
            3. (The set of obstacles imposed by the boundary of the lane) ---> not a duty of this function, but some other function
            4. (The set of available waypoints should be always to the right of the ego vehicle in ego frame) --->
                not a duty of this function though
        """
        # TODO: Below lists the procedure of the function
        '''
            1. Use the center line to generate key interest way points as final destinations on center line
            2. Use the normal of the center line to generate tiles of polygons the fill the lines
            3. (Use the Shapely library to serve as a drivable area boundary check, basically, if a point
               in the search falls within the drivable area and without interfering with the obstacle it is
               a drivable area and can be included, otherwise if it falls out of any polygons or it is too 
               close to the obstacle it is not drivable) ---> not a duty of this function, but some other function
               the "buffer" method can blow the polygon just a bit to include all boundary conditions
               the "disjoint" method can also be used to check if an obstacle is contained or overlaps the lane
            4. Use Shapely's minimal_rotated_rectangle to get the minimal rectangle that encloses the polygon
        '''
        rasterizedOptimizedLanesAroundTraj = copy.deepcopy(optimizedLanesAroundTraj)
        rasterizedOptimizedLanesAroundTraj["processedLane"] = {}
        uniqueLaneIDs = rasterizedOptimizedLanesAroundTraj["uniqueLaneIDs"]
        for onelaneID in uniqueLaneIDs:
            centerLine = self.argoverseMap.get_lane_segment_centerline(onelaneID, self.cityName)
            denseCenterLine = interp_arc(500, centerLine[:, 0], centerLine[:, 1])
            accuDist = 0
            lanePOI = []
            leftSide, rightSide, leftSideBound, rightSideBound = self._calculateOutReachfromNormal(denseCenterLine, 0,
                                                                                                   laneWidth)
            laneRasterizedCoords = {"leftCoord": [leftSide], "rightCoord": [rightSide],
                                    "leftBoundCoord": [leftSideBound], "rightBoundCoord": [rightSideBound]}
            passPOI = False
            for idx in range(1, len(denseCenterLine)):
                accuDist += np.linalg.norm(denseCenterLine[idx] - denseCenterLine[idx - 1])
                if accuDist >= travelDist / 2 and not passPOI:
                    lanePOI.append(copy.deepcopy(denseCenterLine[idx]))
                    passPOI = True
                elif accuDist >= travelDist:
                    leftSide, rightSide, leftSideBound, rightSideBound = self._calculateOutReachfromNormal(
                        denseCenterLine, idx, laneWidth)
                    laneRasterizedCoords["leftCoord"].append(copy.deepcopy(leftSide))
                    laneRasterizedCoords["rightCoord"].append(copy.deepcopy(rightSide))
                    laneRasterizedCoords["leftBoundCoord"].append(copy.deepcopy(leftSideBound))
                    laneRasterizedCoords["rightBoundCoord"].append(copy.deepcopy(rightSideBound))
                    accuDist = 0
                    passPOI = False
                    continue
                # finally if it is the end point and it has not been included
                # in the elif condition (because otherwise we will skip it)
                if idx == len(denseCenterLine) - 1:
                    leftSide, rightSide, leftSideBound, rightSideBound = self._calculateOutReachfromNormal(
                        denseCenterLine, idx, laneWidth)
                    laneRasterizedCoords["leftCoord"].append(copy.deepcopy(leftSide))
                    laneRasterizedCoords["rightCoord"].append(copy.deepcopy(rightSide))
                    laneRasterizedCoords["leftBoundCoord"].append(copy.deepcopy(leftSideBound))
                    laneRasterizedCoords["rightBoundCoord"].append(copy.deepcopy(rightSideBound))

            laneRasterizedCoords["leftCoord"] = np.array(laneRasterizedCoords["leftCoord"])
            laneRasterizedCoords["rightCoord"] = np.array(laneRasterizedCoords["rightCoord"])
            laneRasterizedCoords["leftBoundCoord"] = np.array(laneRasterizedCoords["leftBoundCoord"])
            laneRasterizedCoords["rightBoundCoord"] = np.array(laneRasterizedCoords["rightBoundCoord"])
            lanePOI = np.array(lanePOI)
            # convert all relevant polygon to shapely polygon class
            assert len(laneRasterizedCoords["leftCoord"]) == len(laneRasterizedCoords["rightCoord"])
            assert len(laneRasterizedCoords["leftCoord"]) == len(laneRasterizedCoords["leftBoundCoord"])
            assert len(laneRasterizedCoords["rightCoord"]) == len(laneRasterizedCoords["rightBoundCoord"])
            rasterizedSurface = []
            polygonLeftLaneBound = []
            polygonRightLaneBound = []
            for idx in range(len(laneRasterizedCoords["leftCoord"]) - 1):
                coord0 = laneRasterizedCoords["leftCoord"][idx]
                coord1 = laneRasterizedCoords["leftCoord"][idx + 1]
                coord2 = laneRasterizedCoords["rightCoord"][idx + 1]
                coord3 = laneRasterizedCoords["rightCoord"][idx]
                rasterGrid = geometry.Polygon([coord0, coord1, coord2, coord3])
                rasterGrid = rasterGrid.buffer(0.1)
                rasterizedSurface.append(rasterGrid)

                # get left boundary as an rectangle obstacle
                coord0 = laneRasterizedCoords["leftBoundCoord"][idx]
                coord1 = laneRasterizedCoords["leftBoundCoord"][idx + 1]
                coord2 = laneRasterizedCoords["leftCoord"][idx + 1]
                coord3 = laneRasterizedCoords["leftCoord"][idx]
                rasterGrid = geometry.Polygon([coord0, coord1, coord2, coord3])
                rasterGrid = rasterGrid.buffer(0.1)
                polygonLeftLaneBound.append(rasterGrid)

                # get right boundary as an rectangle obstacle
                coord0 = laneRasterizedCoords["rightBoundCoord"][idx]
                coord1 = laneRasterizedCoords["rightBoundCoord"][idx + 1]
                coord2 = laneRasterizedCoords["rightCoord"][idx + 1]
                coord3 = laneRasterizedCoords["rightCoord"][idx]
                rasterGrid = geometry.Polygon([coord0, coord1, coord2, coord3])
                rasterGrid = rasterGrid.buffer(0.1)
                polygonRightLaneBound.append(rasterGrid)

            # the very last step, the closing boundary of the surface
            _, _, coord1, coord2 = self._calculateOutReachfromNormal(denseCenterLine, 0, laneWidth)
            _, _, coord0, coord3 = self._calculateOutReachfromNormal(denseCenterLine, 1, laneWidth)
            rasterGrid = geometry.Polygon([coord0, coord1, coord2, coord3])
            rasterGrid = rasterGrid.buffer(0.3)
            polygonRearLaneBound = rasterGrid

            _, _, coord1, coord2 = self._calculateOutReachfromNormal(denseCenterLine, -1, laneWidth)
            _, _, coord0, coord3 = self._calculateOutReachfromNormal(denseCenterLine, -2, laneWidth)
            rasterGrid = geometry.Polygon([coord0, coord1, coord2, coord3])
            rasterGrid = rasterGrid.buffer(0.3)
            polygonFrontLaneBound = rasterGrid

            rasterizedOptimizedLanesAroundTraj["processedLane"][onelaneID] = {
                "lanePOI": lanePOI,
                "laneRasterizedCoords": laneRasterizedCoords,
                "rasterizedSurface": rasterizedSurface,
                "polygonRightLaneBound": polygonRightLaneBound,
                "polygonLeftLaneBound": polygonLeftLaneBound,
                "polygonRearLaneBound": polygonRearLaneBound,
                "polygonFrontLaneBound": polygonFrontLaneBound
            }
            if visualize:
                # plot rasterized lane surfaces and polygons
                print("Rasterized lane ID:", onelaneID)
                for surface, leftBound, rightBound in zip(rasterizedSurface, polygonLeftLaneBound,
                                                          polygonRightLaneBound):
                    xs, ys = surface.exterior.xy
                    plt.fill(xs, ys, alpha=0.1, fc='blue', ec='None')
                    xs, ys = leftBound.exterior.xy
                    plt.fill(xs, ys, alpha=0.9, fc='green', ec='None')
                    xs, ys = rightBound.exterior.xy
                    plt.fill(xs, ys, alpha=0.9, fc='orange', ec='None')
                xs, ys = polygonRearLaneBound.exterior.xy
                plt.fill(xs, ys, alpha=0.4, fc='purple', ec='None')
                xs, ys = polygonFrontLaneBound.exterior.xy
                plt.fill(xs, ys, alpha=0.9, fc='red', ec='None')
                plt.plot(denseCenterLine[:, 0], denseCenterLine[:, 1], color='blue')
                plt.scatter(lanePOI[:, 0], lanePOI[:, 1], marker='o', color='red')
        if visualize:
            plt.show()
        self.rasterizedOptimizedLanesAroundTraj = copy.deepcopy(rasterizedOptimizedLanesAroundTraj)
        return rasterizedOptimizedLanesAroundTraj

    def getFirstEgoPose(self):
        assert self.listofTimestamps and len(self.listofTimestamps)
        assert self.logEgoPoseDict and len(self.logEgoPoseDict)
        firstTimestamp = self.listofTimestamps[0]
        return self.logEgoPoseDict[self.currentTrajID][firstTimestamp]

    def setCurrentTrajID(self, trajID):
        self.currentTrajID = trajID

    def getCurrentTraj(self):
        return self.currentTrajID

    def getListofObstaclesIncLaneBoundaries(self, rasterizedOptimizedLanesAroundTraj, onlyVehicles=False):
        """
        The function gets the list of obstacles including the outer lane boundaries; lane boundaries
        they are not tight to a particular timestamp, but other dynamic obstacles are. Two lists of
        dynamic obstacles, one includes off lane obstacles other one includes on lane obstacles.
        """
        def _objectInLane(rasterizedLanes, objecttoTest):
            bboxWorldCoord = objecttoTest.bbox_city_fr
            bboxPolygon = geometry.Polygon([bboxWorldCoord[0], bboxWorldCoord[1], bboxWorldCoord[3], bboxWorldCoord[2]])
            for uniqueID in rasterizedLanes["uniqueLaneIDs"]:
                rasterizedSurfaces = rasterizedLanes["processedLane"][uniqueID]["rasterizedSurface"]
                for laneSurface in rasterizedSurfaces:
                    if not laneSurface.disjoint(bboxPolygon):
                        return True
            return False
        processedLaneID = {"left": set(), "right": set()}
        retListofObstacles = {"dynamicObstacles": {}, "laneBoundaries": {}}
        leftLaneBoundaries = []
        rightLaneBoundaries = []
        for timestamp in self.listofTimestamps:
            retListofObstacles["dynamicObstacles"][timestamp] = {}
            objects = self.logTimestampDict[self.currentTrajID][timestamp]
            allOccluded = True
            offLaneObstacles = []
            onLaneObstacles = []
            for oneObject in objects:
                if oneObject.occlusion_val != IS_OCCLUDED_FLAG:
                    allOccluded = False
                    break
            if not allOccluded:
                for i, oneObject in enumerate(objects):
                    if oneObject.occlusion_val != IS_OCCLUDED_FLAG:
                        if onlyVehicles and "VEHICLE" not in oneObject.obj_class_str:
                            continue
                        if _objectInLane(rasterizedOptimizedLanesAroundTraj, oneObject):
                            onLaneObstacles.append(copy.deepcopy(oneObject))
                        else:
                            offLaneObstacles.append(copy.deepcopy(oneObject))
            retListofObstacles["dynamicObstacles"][timestamp]["offLaneObstacles"] = offLaneObstacles
            retListofObstacles["dynamicObstacles"][timestamp]["onLaneObstacles"] = onLaneObstacles

            # process lanes and lane boundaries
            leftMostLaneID = rasterizedOptimizedLanesAroundTraj[timestamp]["leftMostLaneID"]
            rightMostLaneID = rasterizedOptimizedLanesAroundTraj[timestamp]["rightMostLaneID"]
            egoLaneID = rasterizedOptimizedLanesAroundTraj[timestamp]["egoLaneID"]
            if leftMostLaneID == -1:
                leftMostLaneID = egoLaneID
            if rightMostLaneID == -1:
                rightMostLaneID = egoLaneID
            leftBoundaryPoly = copy.deepcopy(rasterizedOptimizedLanesAroundTraj["processedLane"][leftMostLaneID]["polygonLeftLaneBound"])
            rightBoundaryPoly = copy.deepcopy(rasterizedOptimizedLanesAroundTraj["processedLane"][rightMostLaneID]["polygonRightLaneBound"])

            # get rid of boundaries that entirely overlaps the rasterized surface
            leftBoundaryPolyNew = list()
            rightBoundaryPolyNew = list()
            for idx in range(len(leftBoundaryPoly)):
                notDisjoint = False
                for oneLaneID in rasterizedOptimizedLanesAroundTraj["processedLane"]:
                    if oneLaneID == leftMostLaneID or oneLaneID == rightMostLaneID:
                        continue
                    rasterizedSurface = rasterizedOptimizedLanesAroundTraj["processedLane"][oneLaneID]["rasterizedSurface"]
                    for surface in rasterizedSurface:
                        area = surface.intersection(leftBoundaryPoly[idx]).area/leftBoundaryPoly[idx].area
                        if area > 0.7:
                            print("Left boundary idx: {} overlaps with lane surface {} largely at {}%".format(idx, oneLaneID, area))
                            notDisjoint = True
                            break
                    if notDisjoint:
                        break
                if not notDisjoint:
                    leftBoundaryPolyNew.append(leftBoundaryPoly[idx])
            for idx in range(len(rightBoundaryPoly)):
                notDisjoint = False
                for oneLaneID in rasterizedOptimizedLanesAroundTraj["processedLane"]:
                    if oneLaneID == leftMostLaneID or oneLaneID == rightMostLaneID:
                        continue
                    rasterizedSurface = rasterizedOptimizedLanesAroundTraj["processedLane"][oneLaneID]["rasterizedSurface"]
                    notDisjoint = False
                    for surface in rasterizedSurface:
                        area = surface.intersection(rightBoundaryPoly[idx]).area/rightBoundaryPoly[idx].area
                        if area > 0.7:
                            print("Right boundary idx: {} overlaps with lane surface {} largely at {}%".format(idx, oneLaneID, area))
                            notDisjoint = True
                            break
                    if notDisjoint:
                        break
                if not notDisjoint:
                    rightBoundaryPolyNew.append(rightBoundaryPoly[idx])
            if leftMostLaneID not in processedLaneID["left"]:
                leftLaneBoundaries.extend(leftBoundaryPolyNew)
                processedLaneID["left"].add(leftMostLaneID)
            if rightMostLaneID not in processedLaneID["right"]:
                rightLaneBoundaries.extend(rightBoundaryPolyNew)
                processedLaneID["right"].add(rightMostLaneID)
        retListofObstacles["laneBoundaries"]["leftLaneBoundaries"] = leftLaneBoundaries
        retListofObstacles["laneBoundaries"]["rightLaneBoundaries"] = rightLaneBoundaries

        frontLaneBoundaries = []
        rearLaneBoundaries = []

        for oneLaneID in rasterizedOptimizedLanesAroundTraj["processedLane"]:
            canInclude = True
            rearLaneBound = rasterizedOptimizedLanesAroundTraj["processedLane"][oneLaneID]["polygonRearLaneBound"]
            for referenceLaneID in rasterizedOptimizedLanesAroundTraj["processedLane"]:
                if referenceLaneID != oneLaneID:
                    refFrontLaneBound = rasterizedOptimizedLanesAroundTraj["processedLane"][referenceLaneID][
                        "polygonFrontLaneBound"]
                    # plt.figure()
                    # xs, ys = rearLaneBound.exterior.xy
                    # rxs, rys = refFrontLaneBound.exterior.xy
                    # plt.fill(xs, ys, alpha=0.5, fc='purple', ec='None')
                    # plt.fill(rxs, rys, alpha=0.5, fc='red', ec='None')
                    # plt.show()
                    area = rearLaneBound.intersection(refFrontLaneBound).area/rearLaneBound.area
                    if area > 0.7:
                        canInclude = False
                        break
            if canInclude:
                rearLaneBoundaries.append(copy.deepcopy(rearLaneBound))
            canInclude = True
            frontLaneBound = rasterizedOptimizedLanesAroundTraj["processedLane"][oneLaneID]["polygonFrontLaneBound"]
            for referenceLaneID in rasterizedOptimizedLanesAroundTraj["processedLane"]:
                if referenceLaneID != oneLaneID:
                    refRearLaneBound = rasterizedOptimizedLanesAroundTraj["processedLane"][referenceLaneID][
                        "polygonRearLaneBound"]
                    area = frontLaneBound.intersection(refRearLaneBound).area / frontLaneBound.area
                    if area > 0.7:
                        canInclude = False
                        break
            if canInclude:
                frontLaneBoundaries.append(copy.deepcopy(frontLaneBound))

        retListofObstacles["laneBoundaries"]["frontLaneBoundaries"] = frontLaneBoundaries
        retListofObstacles["laneBoundaries"]["rearLaneBoundaries"] = rearLaneBoundaries
        return retListofObstacles

    def visualizeOccupancyGridOneTimestamp(self, timestamp, refTimestamp,
                                           rasterizedOptimizedLanesAroundTraj,
                                           listofObstacles, savedir,
                                           userTranslate, userRotate,
                                           visRange=70):
        r = R.from_matrix(userRotate)
        angle = r.as_euler('zxy', degrees=True)[0]  # rotation angle centered around z

        # visualize lanes, boundaries, and POI
        fig, ax = plt.subplots(figsize=(3.5, 1.75))
        for laneID in rasterizedOptimizedLanesAroundTraj["uniqueLaneIDs"]:
            rasterizedSurface = copy.deepcopy(
                rasterizedOptimizedLanesAroundTraj["processedLane"][laneID]["rasterizedSurface"])
            lanePOI = copy.deepcopy(rasterizedOptimizedLanesAroundTraj["processedLane"][laneID]["lanePOI"])
            if len(lanePOI):
                lanePOI[:, 0] = lanePOI[:, 0] + userTranslate[0]
                lanePOI[:, 1] = lanePOI[:, 1] + userTranslate[1]
                lanePOI = lanePOI @ userRotate[:2, :2].T
                plt.plot(lanePOI[:, 0], lanePOI[:, 1], '.r', markersize=2)
            for surface in rasterizedSurface:
                surface = translate(surface, xoff=userTranslate[0], yoff=userTranslate[1])
                surface = rotate(surface, angle=angle, origin=(0, 0))
                xs, ys = surface.exterior.xy
                plt.fill(xs, ys, alpha=1.0, fc='#e0e0e0', ec='None')
        leftLaneBoundaries = listofObstacles["laneBoundaries"]["leftLaneBoundaries"]
        for boundary in leftLaneBoundaries:
            boundary = translate(boundary, xoff=userTranslate[0], yoff=userTranslate[1])
            boundary = rotate(boundary, angle=angle, origin=(0, 0))
            xs, ys = boundary.exterior.xy
            plt.fill(xs, ys, fc='#4d4d4d', ec='None')
        rightLaneBoundaries = listofObstacles["laneBoundaries"]["rightLaneBoundaries"]
        for boundary in rightLaneBoundaries:
            boundary = translate(boundary, xoff=userTranslate[0], yoff=userTranslate[1])
            boundary = rotate(boundary, angle=angle, origin=(0, 0))
            xs, ys = boundary.exterior.xy
            plt.fill(xs, ys, fc='#4d4d4d', ec='None')
        frontLaneBoundaries = listofObstacles["laneBoundaries"]["frontLaneBoundaries"]
        for boundary in frontLaneBoundaries:
            boundary = translate(boundary, xoff=userTranslate[0], yoff=userTranslate[1])
            boundary = rotate(boundary, angle=angle, origin=(0, 0))
            xs, ys = boundary.exterior.xy
            plt.fill(xs, ys, fc='#4d4d4d', ec='None')
        rearLaneBoundaries = listofObstacles["laneBoundaries"]["rearLaneBoundaries"]
        for boundary in rearLaneBoundaries:
            boundary = translate(boundary, xoff=userTranslate[0], yoff=userTranslate[1])
            boundary = rotate(boundary, angle=angle, origin=(0, 0))
            xs, ys = boundary.exterior.xy
            plt.fill(xs, ys, fc='#4d4d4d', ec='None')

        # visualize obstacles
        assert timestamp in self.listofTimestamps
        dynamicObstacles = listofObstacles["dynamicObstacles"][timestamp]
        for obstacle in dynamicObstacles["onLaneObstacles"]:
            bboxWorldCoord = obstacle.bbox_city_fr
            bboxPolygon = geometry.Polygon([bboxWorldCoord[0], bboxWorldCoord[1], bboxWorldCoord[3], bboxWorldCoord[2]])
            bboxPolygon = translate(bboxPolygon, xoff=userTranslate[0], yoff=userTranslate[1])
            bboxPolygon = rotate(bboxPolygon, angle=angle, origin=(0, 0))
            xs, ys = bboxPolygon.exterior.xy
            plt.fill(xs, ys, alpha=1.0, fc='#b2182b', ec='#b2182b')
        for obstacle in dynamicObstacles["offLaneObstacles"]:
            bboxWorldCoord = obstacle.bbox_city_fr
            bboxPolygon = geometry.Polygon([bboxWorldCoord[0], bboxWorldCoord[1], bboxWorldCoord[3], bboxWorldCoord[2]])
            bboxPolygon = translate(bboxPolygon, xoff=userTranslate[0], yoff=userTranslate[1])
            bboxPolygon = rotate(bboxPolygon, angle=angle, origin=(0, 0))
            xs, ys = bboxPolygon.exterior.xy
            plt.fill(xs, ys, alpha=1.0, fc='#ef8a62', ec='#ef8a62')

        poseCitytoEgo = self.logEgoPoseDict[self.currentTrajID][timestamp]
        egoTranslation = np.array(poseCitytoEgo["translation"])
        plt.plot(egoTranslation[0] + userTranslate[0], egoTranslation[1] + userTranslate[1], "vc", markersize=2)

        # draw and save
        poseCitytoEgo = self.logEgoPoseDict[self.currentTrajID][refTimestamp]
        refEgoTranslation = np.array(poseCitytoEgo["translation"])
        plt.xlim(
            [refEgoTranslation[0] - visRange + userTranslate[0], refEgoTranslation[0] + visRange + userTranslate[0]])
        plt.ylim(
            [refEgoTranslation[1] - visRange / 2 + userTranslate[1], refEgoTranslation[1] + visRange / 2 + userTranslate[1]])
        ax.axis("off")
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig("{}/{}.pdf".format(savedir, timestamp))
        plt.clf()
        plt.cla()
        plt.close('all')

    def visualizeOccupancyGridXYOneTimestamp(self, timestamp, refTimestamp,
                                             rasterizedOptimizedLanesAroundTraj,
                                             listofObstacles, savedir,
                                             userTranslate, userRotate,
                                             visRange=80):
        r = R.from_matrix(userRotate)
        angle = r.as_euler('zxy', degrees=True)[0]  # rotation angle centered around z

        # visualize lanes, boundaries, and POI
        plt.figure(figsize=(9, 9), dpi=100)
        for laneID in rasterizedOptimizedLanesAroundTraj["uniqueLaneIDs"]:
            rasterizedSurface = copy.deepcopy(
                rasterizedOptimizedLanesAroundTraj["processedLane"][laneID]["rasterizedSurface"])
            lanePOI = copy.deepcopy(rasterizedOptimizedLanesAroundTraj["processedLane"][laneID]["lanePOI"])
            if len(lanePOI):
                lanePOI[:, 0] = lanePOI[:, 0] + userTranslate[0]
                lanePOI[:, 1] = lanePOI[:, 1] + userTranslate[1]
                lanePOI = lanePOI @ userRotate[:2, :2].T
                plt.scatter(lanePOI[:, 0], lanePOI[:, 1], marker='.', color='c')
            for surface in rasterizedSurface:
                surface = translate(surface, xoff=userTranslate[0], yoff=userTranslate[1])
                surface = rotate(surface, angle=angle, origin=(0, 0))
                xs, ys = surface.exterior.xy
                xmax, xmin = np.max(xs), np.min(xs)
                ymax, ymin = np.max(ys), np.min(ys)
                xs = np.array([xmin, xmin, xmax, xmax])
                ys = np.array([ymin, ymax, ymax, ymin])
                plt.fill(xs, ys, alpha=0.2, fc='blue', ec='None')
        leftLaneBoundaries = listofObstacles["laneBoundaries"]["leftLaneBoundaries"]
        for boundary in leftLaneBoundaries:
            boundary = translate(boundary, xoff=userTranslate[0], yoff=userTranslate[1])
            boundary = rotate(boundary, angle=angle, origin=(0, 0))
            xs, ys = boundary.exterior.xy
            xmax, xmin = np.max(xs), np.min(xs)
            ymax, ymin = np.max(ys), np.min(ys)
            xs = np.array([xmin, xmin, xmax, xmax])
            ys = np.array([ymin, ymax, ymax, ymin])
            plt.fill(xs, ys, alpha=0.9, fc='purple', ec='None')
        rightLaneBoundaries = listofObstacles["laneBoundaries"]["rightLaneBoundaries"]
        for boundary in rightLaneBoundaries:
            boundary = translate(boundary, xoff=userTranslate[0], yoff=userTranslate[1])
            boundary = rotate(boundary, angle=angle, origin=(0, 0))
            xs, ys = boundary.exterior.xy
            xmax, xmin = np.max(xs), np.min(xs)
            ymax, ymin = np.max(ys), np.min(ys)
            xs = np.array([xmin, xmin, xmax, xmax])
            ys = np.array([ymin, ymax, ymax, ymin])
            plt.fill(xs, ys, alpha=0.9, fc='orange', ec='None')
        frontLaneBoundaries = listofObstacles["laneBoundaries"]["frontLaneBoundaries"]
        for boundary in frontLaneBoundaries:
            boundary = translate(boundary, xoff=userTranslate[0], yoff=userTranslate[1])
            boundary = rotate(boundary, angle=angle, origin=(0, 0))
            xs, ys = boundary.exterior.xy
            xmax, xmin = np.max(xs), np.min(xs)
            ymax, ymin = np.max(ys), np.min(ys)
            xs = np.array([xmin, xmin, xmax, xmax])
            ys = np.array([ymin, ymax, ymax, ymin])
            plt.fill(xs, ys, alpha=0.5, fc='red', ec='None')
        rearLaneBoundaries = listofObstacles["laneBoundaries"]["rearLaneBoundaries"]
        for boundary in rearLaneBoundaries:
            boundary = translate(boundary, xoff=userTranslate[0], yoff=userTranslate[1])
            boundary = rotate(boundary, angle=angle, origin=(0, 0))
            xs, ys = boundary.exterior.xy
            xmax, xmin = np.max(xs), np.min(xs)
            ymax, ymin = np.max(ys), np.min(ys)
            xs = np.array([xmin, xmin, xmax, xmax])
            ys = np.array([ymin, ymax, ymax, ymin])
            plt.fill(xs, ys, alpha=0.5, fc='red', ec='None')

        # visualize obstacles
        assert timestamp in self.listofTimestamps
        dynamicObstacles = listofObstacles["dynamicObstacles"][timestamp]
        for obstacle in dynamicObstacles["onLaneObstacles"]:
            bboxWorldCoord = obstacle.bbox_city_fr
            bboxPolygon = geometry.Polygon([bboxWorldCoord[0], bboxWorldCoord[1], bboxWorldCoord[3], bboxWorldCoord[2]])
            bboxPolygon = translate(bboxPolygon, xoff=userTranslate[0], yoff=userTranslate[1])
            bboxPolygon = rotate(bboxPolygon, angle=angle, origin=(0, 0))
            xs, ys = bboxPolygon.exterior.xy
            xmax, xmin = np.max(xs), np.min(xs)
            ymax, ymin = np.max(ys), np.min(ys)
            xs = np.array([xmin, xmin, xmax, xmax])
            ys = np.array([ymin, ymax, ymax, ymin])
            plt.fill(xs, ys, alpha=1.0, fc='green', ec='green')
        for obstacle in dynamicObstacles["offLaneObstacles"]:
            bboxWorldCoord = obstacle.bbox_city_fr
            bboxPolygon = geometry.Polygon([bboxWorldCoord[0], bboxWorldCoord[1], bboxWorldCoord[3], bboxWorldCoord[2]])
            bboxPolygon = translate(bboxPolygon, xoff=userTranslate[0], yoff=userTranslate[1])
            bboxPolygon = rotate(bboxPolygon, angle=angle, origin=(0, 0))
            xs, ys = bboxPolygon.exterior.xy
            xmax, xmin = np.max(xs), np.min(xs)
            ymax, ymin = np.max(ys), np.min(ys)
            xs = np.array([xmin, xmin, xmax, xmax])
            ys = np.array([ymin, ymax, ymax, ymin])
            plt.fill(xs, ys, alpha=1.0, fc='red', ec='red')
        poseCitytoEgo = self.logEgoPoseDict[self.currentTrajID][timestamp]
        egoTranslation = np.array(poseCitytoEgo["translation"])
        plt.scatter(egoTranslation[0] + userTranslate[0], egoTranslation[1] + userTranslate[1], marker="o",
                    color="black")

        # draw and save
        poseCitytoEgo = self.logEgoPoseDict[self.currentTrajID][refTimestamp]
        refEgoTranslation = np.array(poseCitytoEgo["translation"])
        plt.xlim(
            [refEgoTranslation[0] - visRange + userTranslate[0], refEgoTranslation[0] + visRange + userTranslate[0]])
        plt.ylim(
            [refEgoTranslation[1] - visRange + userTranslate[1], refEgoTranslation[1] + visRange + userTranslate[1]])
        plt.savefig("{}/{}.jpg".format(savedir, timestamp))
        plt.clf()
        plt.cla()
        plt.close('all')

    def visualizeOccupancyGrid(self, rasterizedOptimizedLanesAroundTraj, listofObstacles, savedir, egoFrame=False):
        """
        The function visualize the occupancy grid and save the visualization to savedir
        """
        print("Visualizing in original polygon coordinates and pure X-Y coordinates.")
        for timestamp in self.listofTimestamps:
            if not egoFrame:
                userTranslate = np.array([0, 0])
                userRotate = np.eye(3)
                self.visualizeOccupancyGridOneTimestamp(timestamp, self.listofTimestamps[0],
                                                        rasterizedOptimizedLanesAroundTraj,
                                                        listofObstacles,
                                                        os.path.join(savedir, "origin"),
                                                        userTranslate=userTranslate, userRotate=userRotate)
                # self.visualizeOccupancyGridXYOneTimestamp(timestamp, self.listofTimestamps[0],
                #                                           rasterizedOptimizedLanesAroundTraj,
                #                                           listofObstacles,
                #                                           os.path.join(savedir, "xy"),
                #                                           userTranslate=userTranslate, userRotate=userRotate)
            else:
                poseCitytoEgo = self.logEgoPoseDict[self.currentTrajID][timestamp]
                userTranslate = -np.array(poseCitytoEgo["translation"][0:2])
                userRotate = poseCitytoEgo["rotation"].T
                self.visualizeOccupancyGridOneTimestamp(timestamp, timestamp,
                                                        rasterizedOptimizedLanesAroundTraj,
                                                        listofObstacles,
                                                        os.path.join(savedir, "origin"),
                                                        userTranslate=userTranslate, userRotate=userRotate)
                # self.visualizeOccupancyGridXYOneTimestamp(timestamp, timestamp,
                #                                           rasterizedOptimizedLanesAroundTraj,
                #                                           listofObstacles,
                #                                           os.path.join(savedir, "xy"),
                #                                           userTranslate=userTranslate, userRotate=userRotate)

    def buildOccupancyGrid(self, visualize=None):
        """
        The function builds the occupancy grid at each time stamp as a list of obstacles including
        the outer boundaries of the drivable areas.
        """
        print("Building occupancy grid for trajectory ID:", self.currentTrajID, "in city:", self.cityName)
        optimizedLanesAroundTraj = self.getLanesAroundCurrentTraj()
        print("Done getting drivable lanes around trajectory.")
        rasterizedOptimizedLanesAroundTraj = self.getRasterizedLanesDistTraj(optimizedLanesAroundTraj)
        print("Done rasterizing lanes into drivable area with surface and boundaries as obstacles.")
        listofObstacles = self.getListofObstaclesIncLaneBoundaries(rasterizedOptimizedLanesAroundTraj)
        print("Done acquiring list of obstacles including outer lane boundaries.")
        print("Note obstacles are in the form of shapely polygons.")
        if visualize:
            self.visualizeOccupancyGrid(rasterizedOptimizedLanesAroundTraj,
                                        listofObstacles,
                                        savedir=visualize,
                                        egoFrame=True)
            print("Done visualization, saved to {}.".format(visualize))
        # Do translation and rotation at planning time, instead of in current time
        return rasterizedOptimizedLanesAroundTraj, listofObstacles

    def poiPerTimestamp(self, timestamp):
        """
        The function gets next possible point of interest (POI) of a timestamp.
        """
        raise NotImplementedError


# Unit testing of this module on the samples
if __name__ == "__main__":
    basePath = path_configs.BASEPATH
    mapDataPath = os.path.join(basePath, "map_files")
    badTrainingData = []
    subFolder = sys.argv[1]
    logID = sys.argv[2]
    trajDataPath = os.path.join(basePath, "argoverse-tracking", subFolder)
    print("====================== Processing ======================", trajDataPath, logID)
    visualizedOut = os.path.join(basePath, "visualize_occupancy_grid", subFolder, logID)
    if not os.path.exists(visualizedOut):
        os.makedirs(visualizedOut)
        os.makedirs(os.path.join(visualizedOut, "origin"))
        os.makedirs(os.path.join(visualizedOut, "xy"))
    try:
        bevGraph = GenerateBEVGraph(mapDataPath=mapDataPath,
                                    trajDataPath=trajDataPath,
                                    logID=logID)
        rasterizedOptimizedLanesAroundTraj, listofObstacles = bevGraph.buildOccupancyGrid(visualize=visualizedOut)
        del bevGraph
    except Exception as e:
        badTrainingData.append(os.path.join(trajDataPath, logID))
        shutil.rmtree(visualizedOut)
        print(e)
        pass
    plt.clf()
    plt.cla()
    plt.close('all')
    gc.collect()
    print("========================= Done =========================")
    print(badTrainingData)
