# def buildGraph(self, coordFrame, lanesAroundTraj=False, userTrans=None, userRotate=None):
#     """
#     coordFrame: The relative coordinate frame of the objects and lanes, in 'ego' frame everything is ego
#                 vehicle-centered in 'global' frame everything is in absolute coordinates or world coordinates,
#                 in 'user' frame everything is translated and rotated by the user provided 3x3 translation
#                 and rotation matrix.
#     lanesAroundTraj: Lanes around the ego trajectory including both that the ego vehicle traveled and lanes that
#                      are adjacent to the ego lanes but in the same travelling direction, if this is not provided
#                      all lanes within a manhattan distance of the ego vehicle will be shown or considered
#     userTrans, userTRotate: The 1x3 numpy translation and 3x3 rotation matrix
#     """
#     if coordFrame == "ego":
#         for timestamp in self.listofTimestamps:
#             if lanesAroundTraj:
#                 assert self.optimizedLanesAroundTraj
#                 self._buildEgoGraphOneFrame(timestamp,
#                                             setofLanes=self.optimizedLanesAroundTraj["uniqueLaneIDs"])
#             else:
#                 self._buildEgoGraphOneFrame(timestamp)
#     elif coordFrame == "global":
#         for timestamp in self.listofTimestamps:
#             if lanesAroundTraj:
#                 self._buildGlobalGraphOneFrame(timestamp,
#                                                self.listofTimestamps[0],
#                                                setofLanes=self.optimizedLanesAroundTraj["uniqueLaneIDs"])
#             else:
#                 self._buildGlobalGraphOneFrame(timestamp,
#                                                self.listofTimestamps[0])
#     elif coordFrame == "user":
#         assert type(userTrans) == np.ndarray
#         assert type(userRotate) == np.ndarray
#         for timestamp in self.listofTimestamps:
#             if lanesAroundTraj:
#                 assert self.optimizedLanesAroundTraj
#                 self._buildUserGraphOneFrame(timestamp,
#                                              userTrans,
#                                              userRotate,
#                                              setofLanes=self.optimizedLanesAroundTraj["uniqueLaneIDs"])
#             else:
#                 self._buildUserGraphOneFrame(timestamp,
#                                              userTrans,
#                                              userRotate)
#     else:
#         raise Exception('Frame not supported, must be "ego", "user" or "global".')

# def _buildUserGraphOneFrame(self, timestamp, userTrans, userRotate,
#                             range=500, visRange=150, onlyVehicles=False, setofLanes=None):
#     print("Building user-defined graph for timestamp: {}".format(timestamp))
#     listofObstacles = dict()
#     fig = plt.figure(figsize=(11, 11), dpi=120)
#     ax = fig.add_subplot(111)
#
#     # get list of obstacles
#     objects = self.logTimestampDict[self.currentTrajID][timestamp]
#     allOccluded = True
#     for oneObject in objects:
#         if oneObject.occlusion_val != IS_OCCLUDED_FLAG:
#             allOccluded = False
#             break
#     if not allOccluded:
#         for i, oneObject in enumerate(objects):
#             bboxCityFrame = oneObject.bbox_city_fr
#             if oneObject.occlusion_val != IS_OCCLUDED_FLAG:
#                 if onlyVehicles and "VEHICLE" not in oneObject.obj_class_str:
#                     continue
#                 # inverse translation and rotation of city frame point cloud
#                 bboxUserFrame = (bboxCityFrame - userTrans) @ userRotate
#                 listofObstacles[oneObject.track_uuid] = bboxUserFrame
#
#                 # below is just for visualization DEBUG
#                 if VIS:
#                     if "VEHICLE" not in oneObject.obj_class_str:
#                         plot_bbox_2D(ax, bboxUserFrame, (0.0, 0.0, 1.0))
#                     else:
#                         plot_bbox_2D(ax, bboxUserFrame, (0.0, 0.5, 0.1))
#
#     # get list of lanes, if user does not provide which lane to draw, discover by manhattan distance
#     poseCitytoEgo = self.logEgoPoseDict[self.currentTrajID][timestamp]
#     egoTranslation = np.array(poseCitytoEgo["translation"])
#     localLaneDirection, directionConf = self.argoverseMap.get_lane_direction(egoTranslation[0:2], self.cityName)
#     citytoEgoSE3 = SE3(rotation=userRotate, translation=userTrans)
#     if not setofLanes:
#         xmin = egoTranslation[0] - range
#         xmax = egoTranslation[0] + range
#         ymin = egoTranslation[1] - range
#         ymax = egoTranslation[1] + range
#         localLanePolygons = self.argoverseMap.find_local_lane_polygons((xmin, xmax, ymin, ymax), self.cityName)
#         newLocalLanePolygons = list()
#         for localLanePolygon in localLanePolygons:
#             localLanePolygon = translate_polygon(localLanePolygon,
#                                                  copy.deepcopy(-citytoEgoSE3.translation))
#             localLanePolygon = rotate_polygon_about_pt(localLanePolygon,
#                                                        copy.deepcopy(citytoEgoSE3.rotation.T),
#                                                        np.zeros((3,)))
#             newLocalLanePolygons.append(localLanePolygon)
#     else:
#         localLaneIDs = copy.deepcopy(setofLanes)
#         newLocalLanePolygons = list()
#         for localLaneID in localLaneIDs:
#             localLanePolygon = self.argoverseMap.get_lane_segment_polygon(localLaneID, self.cityName)
#             localLanePolygon = translate_polygon(localLanePolygon,
#                                                  copy.deepcopy(-citytoEgoSE3.translation))
#             localLanePolygon = rotate_polygon_about_pt(localLanePolygon,
#                                                        copy.deepcopy(citytoEgoSE3.rotation.T),
#                                                        np.zeros((3,)))
#             newLocalLanePolygons.append(localLanePolygon)
#     localLanePolygons = np.array(newLocalLanePolygons, dtype=object)
#     localLaneDirection = np.dot(citytoEgoSE3.rotation.T[0:2, 0:2], localLaneDirection.reshape((2, 1))).T
#     localLaneDirection = localLaneDirection.squeeze()
#
#     # get ego translation under user translation
#     userEgoTranslation = (np.array(poseCitytoEgo["translation"]) - userTrans) @ userRotate
#
#     if VIS:
#         ax.plot(userEgoTranslation[0], userEgoTranslation[1], "ro", ms=5)
#         dx = localLaneDirection[0] * 10
#         dy = localLaneDirection[1] * 10
#         ax.set_xlim([-visRange, +visRange])
#         ax.set_ylim([-visRange, +visRange])
#         draw_lane_polygons(ax, localLanePolygons)
#         ax.arrow(userEgoTranslation[0], userEgoTranslation[1], dx, dy, color="r", width=0.3, zorder=2)
#         # plt.show()
#         plt.savefig("/home/sheng/pre-phd/riskguard/argodataset/argoverse/visualized_out/user/{}.jpg".format(
#             timestamp))
#         plt.close('all')

# def _buildGlobalGraphOneFrame(self, timestamp, refTimestamp, range=500, visRange=100, onlyVehicles=False,
#                               setofLanes=None):
#     print("Building global graph for timestamp: {}".format(timestamp))
#     listofObstacles = dict()
#     fig = plt.figure(figsize=(11, 11), dpi=120)
#     ax = fig.add_subplot(111)
#
#     # get list of obstacles
#     objects = self.logTimestampDict[self.currentTrajID][timestamp]
#     allOccluded = True
#     for oneObject in objects:
#         if oneObject.occlusion_val != IS_OCCLUDED_FLAG:
#             allOccluded = False
#             break
#     if not allOccluded:
#         for i, oneObject in enumerate(objects):
#             bboxCityFrame = oneObject.bbox_city_fr
#             if oneObject.occlusion_val != IS_OCCLUDED_FLAG:
#                 if onlyVehicles and "VEHICLE" not in oneObject.obj_class_str:
#                     continue
#                 listofObstacles[oneObject.track_uuid] = bboxCityFrame
#
#                 # below is just for visualization DEBUG
#                 if VIS:
#                     if "VEHICLE" not in oneObject.obj_class_str:
#                         plot_bbox_2D(ax, bboxCityFrame, (0.0, 0.0, 1.0))
#                     else:
#                         plot_bbox_2D(ax, bboxCityFrame, (0.0, 0.5, 0.1))
#     # localLanePolygons
#     poseCitytoEgo = self.logEgoPoseDict[self.currentTrajID][timestamp]
#     egoTranslation = np.array(poseCitytoEgo["translation"])
#
#     refPoseCitytoEgo = self.logEgoPoseDict[self.currentTrajID][refTimestamp]
#     refEgoTranslation = np.array(refPoseCitytoEgo["translation"])
#
#     if not setofLanes:
#         xmin = egoTranslation[0] - range
#         xmax = egoTranslation[0] + range
#         ymin = egoTranslation[1] - range
#         ymax = egoTranslation[1] + range
#         localLanePolygons = self.argoverseMap.find_local_lane_polygons((xmin, xmax, ymin, ymax), self.cityName)
#     else:
#         localLanePolygons = list()
#         localLaneIDs = copy.deepcopy(setofLanes)
#         for localLaneID in localLaneIDs:
#             localLanePolygon = self.argoverseMap.get_lane_segment_polygon(localLaneID, self.cityName)
#             localLanePolygons.append(localLanePolygon)
#
#     localLaneDirection, directionConf = self.argoverseMap.get_lane_direction(egoTranslation[0:2], self.cityName)
#     if VIS:
#         ax.plot(egoTranslation[0], egoTranslation[1], "ro", ms=5)
#         dx = localLaneDirection[0] * 10
#         dy = localLaneDirection[1] * 10
#         ax.set_xlim([refEgoTranslation[0] - visRange, refEgoTranslation[0] + visRange])
#         ax.set_ylim([refEgoTranslation[1] - visRange, refEgoTranslation[1] + visRange])
#         draw_lane_polygons(ax, localLanePolygons)
#         ax.arrow(egoTranslation[0], egoTranslation[1], dx, dy, color="r", width=0.3, zorder=2)
#         # plt.show()
#         plt.savefig(
#             "/home/sheng/pre-phd/riskguard/argodataset/argoverse/visualized_out/global/{}.jpg".format(
#                 timestamp))
#         plt.close('all')
#
# def _buildEgoGraphOneFrame(self, timestamp, range=100, visRange=150, onlyVehicles=False, setofLanes=None):
#     print("Building ego graph for timestamp: {}".format(timestamp))
#     listofObstacles = dict()
#     fig = plt.figure(figsize=(11, 11), dpi=120)
#     ax = fig.add_subplot(111)
#
#     # get list of obstacles
#     objects = self.logTimestampDict[self.currentTrajID][timestamp]
#     allOccluded = True
#     for oneObject in objects:
#         if oneObject.occlusion_val != IS_OCCLUDED_FLAG:
#             allOccluded = False
#             break
#     if not allOccluded:
#         for i, oneObject in enumerate(objects):
#             bboxEgoFrame = oneObject.bbox_ego_frame
#             if oneObject.occlusion_val != IS_OCCLUDED_FLAG:
#                 if onlyVehicles and "VEHICLE" not in oneObject.obj_class_str:
#                     continue
#                 listofObstacles[oneObject.track_uuid] = bboxEgoFrame
#                 # below is just for visualization DEBUG
#                 if VIS:
#                     if "VEHICLE" not in oneObject.obj_class_str:
#                         plot_bbox_2D(ax, bboxEgoFrame, (0.0, 0.0, 1.0))
#                     else:
#                         plot_bbox_2D(ax, bboxEgoFrame, (0.0, 0.5, 0.1))
#
#     # localLanePolygons
#     poseCitytoEgo = self.logEgoPoseDict[self.currentTrajID][timestamp]
#     egoTranslation = np.array(poseCitytoEgo["translation"])
#     egoRotation = np.array(poseCitytoEgo["rotation"])
#
#     if not setofLanes:
#         xmin = egoTranslation[0] - range
#         xmax = egoTranslation[0] + range
#         ymin = egoTranslation[1] - range
#         ymax = egoTranslation[1] + range
#         localLanePolygons = self.argoverseMap.find_local_lane_polygons((xmin, xmax, ymin, ymax), self.cityName)
#     else:
#         localLanePolygons = copy.deepcopy(setofLanes)
#     localLaneDirection, directionConf = self.argoverseMap.get_lane_direction(egoTranslation[0:2], self.cityName)
#     citytoEgoSE3 = SE3(rotation=egoRotation, translation=egoTranslation)
#     newLocalLanePolygons = list()
#     for localLanePolygon in localLanePolygons:
#         localLanePolygon = translate_polygon(localLanePolygon,
#                                              copy.deepcopy(-citytoEgoSE3.translation))
#         localLanePolygon = rotate_polygon_about_pt(localLanePolygon,
#                                                    copy.deepcopy(citytoEgoSE3.rotation.T),
#                                                    np.zeros((3,)))
#         newLocalLanePolygons.append(localLanePolygon)
#     localLanePolygons = np.array(newLocalLanePolygons, dtype=object)
#
#     localLaneDirection = np.dot(citytoEgoSE3.rotation.T[0:2, 0:2], localLaneDirection.reshape((2, 1))).T
#     localLaneDirection = localLaneDirection.squeeze()
#     if VIS:
#         ax.plot(0, 0, "ro", ms=5)
#         dx = localLaneDirection[0] * 10
#         dy = localLaneDirection[1] * 10
#         draw_lane_polygons(ax, localLanePolygons)
#         ax.set_xlim([-visRange, +visRange])
#         ax.set_ylim([-visRange, +visRange])
#         ax.arrow(0, 0, dx, dy, color="r", width=0.3, zorder=2)
#         # plt.show()
#         plt.savefig("/home/sheng/pre-phd/riskguard/argodataset/argoverse/visualized_out/ego/{}.jpg".format(
#             timestamp))
#         plt.close('all')

# DEBUG: artificial obstacle
# dynamicObstaclesXY.append([29.052917468489572, 1.97976754443818, 33.785473942740356, 4.5587280157639114])
# dynamicObstaclesPoly.append(geometry.Polygon([(29.052917468489572, 1.97976754443818),
#                                               (29.052917468489572, 4.5587280157639114),
#                                               (33.785473942740356, 4.5587280157639114),
#                                               (29.052917468489572, 4.5587280157639114)]))

# dynamicObstaclesXY.append([38.052917468489572, 1.97976754443818, 42.785473942740356, 4.5587280157639114])
# dynamicObstaclesPoly.append(geometry.Polygon([(38.052917468489572, 1.97976754443818),
#                                               (38.052917468489572, 4.5587280157639114),
#                                               (42.785473942740356, 4.5587280157639114),
#                                               (38.052917468489572, 4.5587280157639114)]))

# dynamicObstaclesXY.append([15.052917468489572, 1.97976754443818,
#                            23.785473942740356, 4.5587280157639114])
# dynamicObstaclesPoly.append(geometry.Polygon([(19.052917468489572, 1.97976754443818),
#                                               (19.052917468489572, 4.5587280157639114),
#                                               (23.785473942740356, 4.5587280157639114),
#                                               (19.052917468489572, 4.5587280157639114)]))


# if self.posUnc != "None":
#     if self.posUnc == "gaussian2DCenter":
#         currentCenter = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2])
#         centerMeanXY = np.array(self.posUncConfig["centerMeanXY"])
#         centerCovariance = np.array(self.posUncConfig["centerCovariance"])
#         sampledCenter = np.random.multivariate_normal(currentCenter + centerMeanXY,
#                                                       centerCovariance, 1)[0]
#         centerDelta = sampledCenter - currentCenter
#         xmin = xmin + centerDelta[0]
#         ymin = ymin + centerDelta[1]
#         xmax = xmax + centerDelta[0]
#         ymax = ymax + centerDelta[1]
#         print("Changing center delta:", centerDelta)
#
#     elif self.posUnc == "gaussian2DCorners":
#         xMinMean = np.array(self.posUncConfig["xMinMean"])
#         yMinMean = np.array(self.posUncConfig["yMinMean"])
#         xMaxMean = np.array(self.posUncConfig["xMaxMean"])
#         yMaxMean = np.array(self.posUncConfig["yMaxMean"])
#         xMinVariance = np.array(self.posUncConfig["xMinVariance"])
#         yMinVariance = np.array(self.posUncConfig["yMinVariance"])
#         xMaxVariance = np.array(self.posUncConfig["xMaxVariance"])
#         yMaxVariance = np.array(self.posUncConfig["yMaxVariance"])
#         xmin = np.random.normal(xMinMean + xmin, np.sqrt(xMinVariance), 1)[0]
#         ymin = np.random.normal(yMinMean + ymin, np.sqrt(yMinVariance), 1)[0]
#         xmax = np.random.normal(xMaxMean + xmax, np.sqrt(xMaxVariance), 1)[0]
#         ymax = np.random.normal(yMaxMean + ymax, np.sqrt(yMaxVariance), 1)[0]
#         print("Changing corner delta:", [xmin, ymin, xmax, ymax])
#     else:
#         raise NotImplementedError
# else:
#     print("Positional uncertainty is not enabled for this run.")


