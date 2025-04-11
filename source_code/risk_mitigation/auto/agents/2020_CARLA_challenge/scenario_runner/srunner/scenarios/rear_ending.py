#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# modified by submission #104

"""
Leading Vehicle Slowing Down:

The scenario realizes a common driving behavior, in which the
user-controlled ego vehicle follows a leading car driving down
a given road. At some point the leading car has to slow down and
finally stop. The ego vehicle has to react accordingly to avoid
a collision. The scenario ends either via a timeout, or if the ego
vehicle stopped close enough to the leading vehicle
"""

import json
import random
import os
import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      KeepVelocity,
                                                                      StopVehicle,
                                                                      WaypointFollower,
                                                                      ChangeAutoPilot)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToVehicle,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance,
                                                                               StandStill)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_waypoint_in_distance


class RearEnding(BasicScenario):
    """
    This class holds everything required for a simple "Follow a leading vehicle"
    scenario involving two vehicles.  (Traffic Scenario 2)

    This is a single ego vehicle scenario
    """

    timeout = 120  # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario

        If randomize is True, the scenario parameters are randomized
        """

        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_stop_in_front_intersection = 20
        self._other_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout

        # scenario configs
        self.class_name = self.__class__.__name__
        self.scene_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                              "..", "scenario_configs", "{}.json".format(self.class_name))
        self.scene_config = {
            "first_vehicle_location": 50,
            "first_vehicle_speed": 50,
            "first_vehicle_location_offset": -80,
            "other_actor_max_brake": 1.0,
            "fourth_vehicle_speed": 30
        }

        # self._second_vehicle_location = 0
        # self._second_vehicle_speed = 70
        self._third_vehicle_location = 0
        self._third_vehicle_speed = 300
        self._fourth_vehicle_location = 0

        # self._other_actor_transform2 = None
        self._other_actor_transform3 = None
        self._other_actor_transform4 = None

        if os.path.isfile(self.scene_config_path):
            with open(self.scene_config_path, "r") as f:
                overwrite_config = json.load(f)
                for key, value in overwrite_config.items():
                    if key in self.scene_config:
                        print("Replacing key {}'s value from {} to {}.".format(key, self.scene_config[key], value))
                        self.scene_config[key] = value
                    else:
                        print("Ignoring key {} because is not configurable.".format(key))
        else:
            print("Path {} not exist, using default configs.".format(os.path.isfile(self.scene_config_path)))

        print("Config for {} scenario is {}".format(self.class_name, self.scene_config))

        super(RearEnding, self).__init__("FollowVehicle",
                                           ego_vehicles,
                                           config,
                                           world,
                                           debug_mode,
                                           criteria_enable=criteria_enable)

        if randomize:
            self._ego_other_distance_start = random.randint(4, 8)

            # Example code how to randomize start location
            # distance = random.randint(20, 80)
            # new_location, _ = get_location_in_distance(self.ego_vehicles[0], distance)
            # waypoint = CarlaDataProvider.get_map().get_waypoint(new_location)
            # waypoint.transform.location.z += 39
            # self.other_actors[0].set_transform(waypoint.transform)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self.scene_config["first_vehicle_location"])
        # second_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        third_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._third_vehicle_location)
        fourth_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._fourth_vehicle_location)
        self._other_actor_transform = carla.Transform(
            carla.Location(first_vehicle_waypoint.transform.location.x - 1.0,
                           first_vehicle_waypoint.transform.location.y + self.scene_config["first_vehicle_location_offset"],
                           first_vehicle_waypoint.transform.location.z + 0.2),
            first_vehicle_waypoint.transform.rotation)
        first_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z - 500),
            self._other_actor_transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.audi.a2',
                                                            first_vehicle_transform)
        first_vehicle.set_simulate_physics(enabled=True)
        self.other_actors.append(first_vehicle)

        # get the transform for the second vehicle
        # self._other_actor_transform2 = carla.Transform(
        #     carla.Location(second_vehicle_waypoint.transform.location.x,
        #                    second_vehicle_waypoint.transform.location.y + 15,
        #                    second_vehicle_waypoint.transform.location.z + 0.1),
        #     second_vehicle_waypoint.transform.rotation)
        # second_vehicle_transform = carla.Transform(
        #     carla.Location(self._other_actor_transform2.location.x,
        #                    self._other_actor_transform2.location.y + 15,
        #                    self._other_actor_transform2.location.z - 500),
        #     self._other_actor_transform2.rotation)
        # second_vehicle = CarlaDataProvider.request_new_actor('vehicle.tesla.model3',
        #                                                      second_vehicle_transform)
        # second_vehicle.set_simulate_physics(enabled=True)
        # self.other_actors.append(second_vehicle)

        # get the transform for the third vehicle
        self._other_actor_transform3 = carla.Transform(
            carla.Location(third_vehicle_waypoint.transform.location.x + 3.2,
                           third_vehicle_waypoint.transform.location.y + 35,
                           third_vehicle_waypoint.transform.location.z + 0.1),
            third_vehicle_waypoint.transform.rotation)
        third_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform3.location.x - 3.2,
                           self._other_actor_transform3.location.y + 35,
                           self._other_actor_transform3.location.z - 500),
            self._other_actor_transform3.rotation)
        third_vehicle = CarlaDataProvider.request_new_actor('vehicle.harley-davidson.low_rider',
                                                            third_vehicle_transform)
        third_vehicle.set_simulate_physics(enabled=True)
        self.other_actors.append(third_vehicle)

        # get the transform for the fourth vehicle
        self._other_actor_transform4 = carla.Transform(
            carla.Location(fourth_vehicle_waypoint.transform.location.x - 3.2,
                           fourth_vehicle_waypoint.transform.location.y + 25,
                           fourth_vehicle_waypoint.transform.location.z + 0.1),
            fourth_vehicle_waypoint.transform.rotation)
        fourth_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform4.location.x - 3.2,
                           self._other_actor_transform4.location.y - 25,
                           self._other_actor_transform4.location.z - 500),
            self._other_actor_transform4.rotation)
        fourth_vehicle = CarlaDataProvider.request_new_actor('vehicle.toyota.prius',
                                                             fourth_vehicle_transform)
        fourth_vehicle.set_simulate_physics(enabled=True)
        self.other_actors.append(fourth_vehicle)

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive until reaching
        the next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # to avoid the other actor blocking traffic, it was spawed elsewhere
        # reset its pose to the required one
        start_transform = py_trees.composites.Parallel("Get all actors",
                                                       policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        start_transform1 = ActorTransformSetter(self.other_actors[0], self._other_actor_transform)
        # start_transform2 = ActorTransformSetter(self.other_actors[1], self._other_actor_transform2)
        start_transform3 = ActorTransformSetter(self.other_actors[1], self._other_actor_transform3)
        start_transform4 = ActorTransformSetter(self.other_actors[2], self._other_actor_transform4)

        start_transform.add_child(start_transform1)
        # start_transform.add_child(start_transform2)
        start_transform.add_child(start_transform3)
        start_transform.add_child(start_transform4)

        # driving_to_next_intersection_first2 = py_trees.composites.Sequence("Start Driving 2")
        driving_to_next_intersection_first3 = py_trees.composites.Sequence("Start Driving 3")
        driving_to_next_intersection_first4 = py_trees.composites.Sequence("Start Driving 4")

        # let the other actor drive the cause a rear end accident
        driving_to_next_intersection = py_trees.composites.Parallel("Driving forward", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        driving_to_next_intersection_first = py_trees.composites.Sequence("Start Driving")
        driving_to_next_intersection_first.add_child(InTriggerDistanceToVehicle(self.other_actors[0],
                                                                                self.ego_vehicles[0],
                                                                                distance=70,
                                                                                name="Distance"))
        driving_to_next_intersection_first.add_child(WaypointFollower(self.other_actors[0], self.scene_config["first_vehicle_speed"]))

        # driving_to_next_intersection_first2.add_child(
        #     ChangeAutoPilot(self.other_actors[1], True, parameters={"max_speed": self._second_vehicle_speed}))
        # driving_to_next_intersection_first2.add_child(
        #     WaypointFollower(self.other_actors[1], self._second_vehicle_speed))

        driving_to_next_intersection_first3.add_child(
            ChangeAutoPilot(self.other_actors[1], True, parameters={"max_speed": self._third_vehicle_speed}))
        driving_to_next_intersection_first3.add_child(WaypointFollower(self.other_actors[1], self._third_vehicle_speed))

        driving_to_next_intersection_first4.add_child(
            ChangeAutoPilot(self.other_actors[2], True, parameters={"max_speed": self.scene_config["fourth_vehicle_speed"]}))
        driving_to_next_intersection_first4.add_child(
            WaypointFollower(self.other_actors[2], self.scene_config["fourth_vehicle_speed"]))


        # construct scenario
        driving_to_next_intersection.add_child(driving_to_next_intersection_first)
        # driving_to_next_intersection.add_child(driving_to_next_intersection_first2)
        driving_to_next_intersection.add_child(driving_to_next_intersection_first3)
        driving_to_next_intersection.add_child(driving_to_next_intersection_first4)

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part = StandStill(self.ego_vehicles[0], name="StandStill", duration=15)
        endcondition.add_child(endcondition_part)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(start_transform)
        sequence.add_child(driving_to_next_intersection)
        sequence.add_child(endcondition)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
