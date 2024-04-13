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
import numpy as np
from shapely import geometry
from shapely.affinity import translate, rotate


def cvctrPrediction(prev1TimeObstacleDict, prev2TimeObstacleDict, currentTime, previousTime1, previousTime2) -> dict:
    # if we cannot get velocity, angular velocity we cannot predict using this model
    if not len(prev2TimeObstacleDict):
        return copy.deepcopy(prev1TimeObstacleDict)
    else:
        timeDelta1 = (currentTime - previousTime1) / 1e9
        timeDelta2 = (previousTime1 - previousTime2) / 1e9

        # for prev2, it could be offLane or onLane
        obstaclesPrev2 = prev2TimeObstacleDict["offLaneObstacles"] + prev2TimeObstacleDict["onLaneObstacles"]

        # first process offLaneObstacles in prev1, which is directly related to the current Time
        offLaneObstacles = list()
        offLaneObstaclesPrev1 = prev1TimeObstacleDict["offLaneObstacles"]
        for o1 in offLaneObstaclesPrev1:
            oId1 = o1['Id']
            found = None
            for o2 in obstaclesPrev2:
                if oId1 == o2['Id']:
                    found = o2
                    break
            currO = dict()
            if found:  # if oId1 can be found in prev2 List
                yaw = o1['rotation'][1]
                vs = np.linalg.norm(o1['location'] - found['location']) / timeDelta2  # linear speed
                ws = (o1['rotation'] - found['rotation']) / timeDelta2  # rotation rate
                currentDimension = o1['dimension']
                currentLocation = o1['location'] + np.array([timeDelta1 * -np.cos(np.deg2rad(yaw)) * vs,
                                                             timeDelta1 * np.sin(np.deg2rad(yaw)) * vs,
                                                             0])
                currentRotation = o1['rotation'] + timeDelta1 * ws
                coord0 = o1["bboxWorldVertices"][0][0:2]
                coord1 = o1["bboxWorldVertices"][2][0:2]
                coord2 = o1["bboxWorldVertices"][6][0:2]
                coord3 = o1["bboxWorldVertices"][4][0:2]
                worldBboxPolygon = geometry.Polygon([coord0, coord1, coord2, coord3])
                worldBboxPolygon = translate(worldBboxPolygon,
                                             xoff=currentLocation[0] - o1['location'][0],
                                             yoff=currentLocation[1] - o1['location'][1])
                worldBboxPolygon = rotate(worldBboxPolygon,
                                          angle=-(currentRotation[1] - o1['rotation'][1]))  # rotation about the center
                worldBboxPolygon = worldBboxPolygon.minimum_rotated_rectangle
                bboxWorldVertices = [
                    np.array([worldBboxPolygon.exterior.xy[0][0], worldBboxPolygon.exterior.xy[1][0], 0]),
                    np.array([0, 0, 0]),
                    np.array([worldBboxPolygon.exterior.xy[0][1], worldBboxPolygon.exterior.xy[1][1], 0]),
                    np.array([0, 0, 0]),
                    np.array([worldBboxPolygon.exterior.xy[0][2], worldBboxPolygon.exterior.xy[1][2], 0]),
                    np.array([0, 0, 0]),
                    np.array([worldBboxPolygon.exterior.xy[0][3], worldBboxPolygon.exterior.xy[1][3], 0]),
                    np.array([0, 0, 0])
                ]
                currO['Id'] = oId1
                currO['dimension'] = currentDimension
                currO['location'] = currentLocation
                currO['rotation'] = currentRotation
                currO['bboxWorldVertices'] = bboxWorldVertices
            else:
                currO = copy.deepcopy(o1)
            offLaneObstacles.append(currO)

        # second process onLaneObstacles
        onLaneObstacles = list()
        onLaneObstaclesPrev1 = prev1TimeObstacleDict["onLaneObstacles"]
        for o1 in onLaneObstaclesPrev1:
            oId1 = o1['Id']
            found = None
            for o2 in obstaclesPrev2:
                if oId1 == o2['Id']:
                    found = o2
                    break
            currO = dict()
            if found:  # if oId1 can be found in prev2 List
                yaw = o1['rotation'][1]
                vs = np.linalg.norm(o1['location'][0:2] - found['location'][0:2]) / timeDelta2  # planer linear speed
                ws = (o1['rotation'] - found['rotation']) / timeDelta2  # rotation rate
                currentDimension = o1['dimension']
                currentLocation = o1['location'] + np.array([timeDelta1 * -np.cos(np.deg2rad(yaw)) * vs,
                                                             timeDelta1 * np.sin(np.deg2rad(yaw)) * vs,
                                                             0])
                currentRotation = o1['rotation'] + timeDelta1 * ws
                coord0 = o1["bboxWorldVertices"][0][0:2]
                coord1 = o1["bboxWorldVertices"][2][0:2]
                coord2 = o1["bboxWorldVertices"][6][0:2]
                coord3 = o1["bboxWorldVertices"][4][0:2]
                worldBboxPolygon = geometry.Polygon([coord0, coord1, coord2, coord3])
                worldBboxPolygon = translate(worldBboxPolygon,
                                             xoff=currentLocation[0] - o1['location'][0],
                                             yoff=currentLocation[1] - o1['location'][1])
                worldBboxPolygon = rotate(worldBboxPolygon,
                                          angle=-(currentRotation[1] - o1['rotation'][1]))
                worldBboxPolygon = worldBboxPolygon.minimum_rotated_rectangle
                bboxWorldVertices = [
                    np.array([worldBboxPolygon.exterior.xy[0][0], worldBboxPolygon.exterior.xy[1][0], 0]),
                    np.array([0, 0, 0]),
                    np.array([worldBboxPolygon.exterior.xy[0][1], worldBboxPolygon.exterior.xy[1][1], 0]),
                    np.array([0, 0, 0]),
                    np.array([worldBboxPolygon.exterior.xy[0][3], worldBboxPolygon.exterior.xy[1][3], 0]),
                    np.array([0, 0, 0]),
                    np.array([worldBboxPolygon.exterior.xy[0][2], worldBboxPolygon.exterior.xy[1][2], 0]),
                    np.array([0, 0, 0])
                ]
                currO['Id'] = oId1
                currO['dimension'] = currentDimension
                currO['location'] = currentLocation
                currO['rotation'] = currentRotation
                currO['bboxWorldVertices'] = bboxWorldVertices
            else:
                currO = copy.deepcopy(o1)
            onLaneObstacles.append(currO)

        currentTimePredictedObstacles = {
            "offLaneObstacles": offLaneObstacles,
            "onLaneObstacles": onLaneObstacles
        }
    return currentTimePredictedObstacles
