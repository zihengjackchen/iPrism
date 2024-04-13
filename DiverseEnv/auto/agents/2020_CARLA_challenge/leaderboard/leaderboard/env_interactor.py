#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function
from ast import arg
from sqlite3 import Timestamp

import traceback
from abc import ABC, abstractmethod
from argparse import Namespace
import time
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys
import torchvision

import carla
import signal
from leaderboard.carla_sim_datautils import DataCollector

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager import ScenarioManager

# import scanario generators
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.scenarios.fi_scenario import FIScenario

from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


sensors_to_icons = {
    'sensor.camera.semantic_segmentation': 'carla_camera',
    'sensor.camera.rgb': 'carla_camera',
    'sensor.lidar.ray_cast': 'carla_lidar',
    'sensor.other.radar': 'carla_radar',
    'sensor.other.gnss': 'carla_gnss',
    'sensor.other.imu': 'carla_imu',
    'sensor.opendrive_map': 'carla_opendrive_map',
    'sensor.speedometer': 'carla_speedometer'
}


class LeaderboardEvaluatorRLEnv(object):
    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 40.0  # in Hz

    def __init__(self, args, statistics_manager, control_log_path=None, gym_environment=None):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.statistics_manager = statistics_manager
        self.sensors = None
        self.sensor_icons = []
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        if gym_environment is None:
            self.client = carla.Client(args.host, int(args.port))
        else:
            self.client = gym_environment.simulator._client
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        if gym_environment is None:
            self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))
        else:
            self.traffic_manager = gym_environment.simulator._traffic_manager

        dist = pkg_resources.get_distribution("carla")
        if dist.version != 'leaderboard':
            if LooseVersion(dist.version) < LooseVersion('0.9.10'):
                raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(
            args.timeout, args.debug > 1, control_log_path=control_log_path, 
            risk_evaluation_mode=args.risk_evaluation_mode,
            risk_pickle_stored=args.risk_pickle_stored,
            traj_pickle_stored=args.traj_pickle_stored,
            inference_model_path=args.inference_model_path,
            inference_config_path=args.inference_config_path,
        )

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Create the agent timer
        self._agent_watchdog = Watchdog(int(float(args.timeout)))
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took too long to setup")
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        # Simulation still running and in synchronous mode?
        if self.manager and self.manager.get_running_status() \
                and hasattr(self, 'world') and self.world:
            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        if self.manager:
            self.manager.cleanup()

        CarlaDataProvider.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

        if hasattr(self, 'statistics_manager') and self.statistics_manager:
            self.statistics_manager.scenario = None

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                             vehicle.transform,
                                                                             vehicle.rolename,
                                                                             color=vehicle.color,
                                                                             actor_category=vehicle.category))

        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None, environment=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """
        if environment is None:
            self.world = self.client.load_world(town)
        else:
            self.world = environment.simulator._world
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(args.trafficManagerPort))

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(args.trafficManagerSeed))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            "This scenario requires to use map {}".format(town))

    def _register_statistics(self, config, checkpoint, entry_status, crash_message=""):
        """
        Computes and saved the simulation statistics
        """
        # register statistics
        current_stats_record = self.statistics_manager.compute_route_statistics(
            config,
            self.manager.scenario_duration_system,
            self.manager.scenario_duration_game,
            crash_message
        )

        print("\033[1m> Registering the route statistics\033[0m")
        self.statistics_manager.save_record(current_stats_record, config.index, checkpoint)
        self.statistics_manager.save_entry_status(entry_status, False, checkpoint)

    def _load_scenario(self, args, config, environment):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        self.crash_message = ""
        self.entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index))
        print("> Setting up the agent\033[0m")

        # Prepare the statistics of the route
        self.statistics_manager.set_route(config.name, config.index)

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            if not args.dual_dup:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config, 0,
                                                                                   preprocessing=args.preprocessing,
                                                                                   preparam=args.preparam)
            else:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config, 0, True)
            self.agent_instance2 = None
            if args.dual_agent:
                if not args.dual_dup:
                    self.agent_instance2 = getattr(self.module_agent, agent_class_name)(args.agent_config, 1,
                                                                                        preprocessing=args.preprocessing,
                                                                                        preparam=args.preparam)
                else:
                    self.agent_instance2 = getattr(self.module_agent, agent_class_name)(args.agent_config, 1, True)
            config.agent = self.agent_instance

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons, args.checkpoint)

            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Agent's sensors were invalid"
            self.entry_status = "Rejected"

            self._register_statistics(config, args.checkpoint, self.entry_status, self.crash_message)
            self._cleanup()
            sys.exit(-1)

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("Could not set up the required agent:")
            print("> {}\n".format(e))
            traceback.print_exc()

            self.crash_message = "Agent couldn't be set up"

            self._register_statistics(config, args.checkpoint, self.entry_status, self.crash_message)
            self._cleanup()
            return

        print("Loading the world")

        # Load the world and the scenario
        try:
            # load the world
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles, environment=environment)

            # collect all static information
            routes = args.routes.split("/")[-1]
            scenarios = args.scenarios.split("/")[-1]
            self.dc = DataCollector(args.sim_data_save, routes + "_" + scenarios+".pkl",
                                    self.world, self.world.get_map())
            self.dc.collect_lane_information(config, travel_dist=4.5)
            self.dc.collect_static_obstacles()
            self.manager.set_data_collector(self.dc)

            self._prepare_ego_vehicles(config.ego_vehicles, False)

            # change to fault injection scenario
            if "fi" in args.routes:
                if environment is not None:
                    self.scenario = FIScenario(world=self.world, config=config, debug_mode=args.debug,
                                               ego_vehicle=environment.simulator._hero)
                else:
                    self.scenario = FIScenario(world=self.world, config=config, debug_mode=args.debug,
                                               ego_vehicle=None)
            else:
                self.scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)

            self.statistics_manager.set_scenario(self.scenario.scenario)

            # Night mode
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in self.scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))

            self.manager.load_scenario(self.scenario,
                                       self.agent_instance,
                                       config.repetition_index,
                                       args.dual_agent,
                                       self.agent_instance2,
                                       args.dual_dup)

        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"

            self._register_statistics(config, args.checkpoint, self.entry_status, self.crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()
            sys.exit(-1)
    
    def _run_scenario(self, args, config):
        print("Running the route")

        # Run the scenario
        try:
            self.manager.run_scenario()

        except AgentError as e:
            # The agent has failed -> stop the route
            print("\nStopping the route, the agent has crashed:")
            print("> {}\n".format(e))
            traceback.print_exc()

            self.crash_message = "Agent crashed"

        except Exception as e:
            print("\nError during the simulation:")
            print("> {}\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"

        # Stop the scenario
        try:
            print("Stopping the route")
            self.manager.stop_scenario()
            self._register_statistics(config, args.checkpoint, self.entry_status, self.crash_message)

            # save data collection results
            self.dc.save_collected_data()

            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            # self.scenario.remove_all_actors()

            self._cleanup()

        except Exception as e:
            print("\nFailed to stop the scenario, the statistics might be empty:")
            print("> {}\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"

        if self.crash_message == "Simulation crashed":
            sys.exit(-1)

    def _step_scenario_getdata(self):
        # print("Stepping the route, get data")

        # get data from the scenario
        try:
            data, timestamp, rewards = self.manager.step_scenario_getdata()
        except AgentError as e:
            # The agent has failed -> stop the route
            print("\nStopping the route, the agent has crashed:")
            print("> {}\n".format(e))
            traceback.print_exc()
            self.crash_message = "Agent crashed"
            data = None
            timestamp = None
            rewards = None

        except Exception as e:
            print("\nError during the simulation:")
            print("> {}\n".format(e))
            traceback.print_exc()
            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"
            data = None
            timestamp = None
            rewards = None
        return data, timestamp, rewards

    def _step_scenario_act(self, timestamp, action):
        # print("Stepping the route, act")

        # Act on the the scenario
        try:
            running = self.manager.step_scenario_act(timestamp, action)
        except AgentError as e:
            # The agent has failed -> stop the route
            print("\nStopping the route, the agent has crashed:")
            print("> {}\n".format(e))
            traceback.print_exc()
            running = False
            self.crash_message = "Agent crashed"
        except Exception as e:
            print("\nError during the simulation:")
            print("> {}\n".format(e))
            traceback.print_exc()
            running = False
            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"
        done = not running
        return done

    def _step_scenario_act_control(self, timestamp, control, environment):
        # print("Stepping the route, act")

        # Act on the the scenario
        try:
            running = self.manager.step_scenario_act_control(timestamp, control, environment)
        except AgentError as e:
            # The agent has failed -> stop the route
            print("\nStopping the route, the agent has crashed:")
            print("> {}\n".format(e))
            traceback.print_exc()
            running = False
            self.crash_message = "Agent crashed"
        except Exception as e:
            print("\nError during the simulation:")
            print("> {}\n".format(e))
            traceback.print_exc()
            running = False
            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"
        done = not running
        return done

    def _stop_scenario(self, args, config):
        print("Simulation done, stopping the scenario")
        # Stop the scenario
        try:
            print("Stopping the route")
            self.manager.stop_scenario()
            self._register_statistics(config, args.checkpoint, self.entry_status, self.crash_message)

            # save data collection results
            self.dc.save_collected_data()

            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            # self.scenario.remove_all_actors()

            self._cleanup()

        except Exception as e:
            print("\nFailed to stop the scenario, the statistics might be empty:")
            print("> {}\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"

        if self.crash_message == "Simulation crashed":
            sys.exit(-1)
        
    def run(self, args):
        """
        Run the challenge mode
        """
        route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)

        if args.resume:
            route_indexer.resume(args.checkpoint)
            self.statistics_manager.resume(args.checkpoint)
        else:
            self.statistics_manager.clear_record(args.checkpoint)
            route_indexer.save_state(args.checkpoint)

        while route_indexer.peek():
            # setup
            config = route_indexer.next()

            # load scenario
            self._load_scenario(args, config)

            # run scenario
            self._run_scenario(args, config)

            route_indexer.save_state(args.checkpoint)

        # save global statistics
        print("Registering the global statistics")
        global_stats_record = self.statistics_manager.compute_global_statistics(route_indexer.total)
        StatisticsManager.save_global_record(global_stats_record, self.sensor_icons, route_indexer.total,
                                             args.checkpoint)

    def prepare_step(self, args, environment):
        """
        Prepare to step the scenario
        """
        self.route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
        if args.resume:
            self.route_indexer.resume(args.checkpoint)
            self.statistics_manager.resume(args.checkpoint)
        else:
            self.statistics_manager.clear_record(args.checkpoint)
            self.route_indexer.save_state(args.checkpoint)
        
        try:
            assert self.route_indexer.peek()
        except AssertionError:
            print("The route is not ready.")
        try:
            assert self.route_indexer.n_routes == 1
        except AssertionError:
            print("Only support running the first route for stepping.")
        
        self.route_indexer_config = self.route_indexer.next()
        self._load_scenario(args, self.route_indexer_config, environment)
    
    def step_sense(self):
        """
        Sense the scenario by getting data
        """
        data, timestamp, rewards = self._step_scenario_getdata()
        return data, timestamp, rewards

    def step_act(self, timestamp, action):
        """
        Run the scenario by acting
        """
        done = self._step_scenario_act(timestamp, action)
        return done

    def step_act_control(self, timestamp, control, environment):
        """
        Run the scenario by acting
        """
        done = self._step_scenario_act_control(timestamp, control, environment)
        return done
    
    def after_step(self, args):
        """
        After the scenario stepping terminiates
        """
        # properly stopping the scenario
        self._stop_scenario(args, self.route_indexer_config)

        # save checkpoint state
        self.route_indexer.save_state(args.checkpoint)
        
        # skip saving global statistics
        print("Registering the global statistics")
        global_stats_record = self.statistics_manager.compute_global_statistics(self.route_indexer.total)
        StatisticsManager.save_global_record(global_stats_record, self.sensor_icons, self.route_indexer.total, args.checkpoint)


class Env(ABC):
    """
    An abstract class that contains pure virtual functions
    run: run the entire scenario until finishes
    step: step the scenario by one tick
    reset: resetting the scenario so that it is ready to be ran again 
    """
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def step_sense(self):
        pass

    @abstractmethod
    def step_act(self, timestamp, action):
        pass

    @abstractmethod
    def reset(self):
        pass


class CarlaRLEnv(Env):
    def __init__(self, config_dict, gym_environment=None):
        self.prepare_step_flag = False
        self.config_dict = config_dict
        self.arguments = Namespace(**self.config_dict)
        self.statistics_manager = StatisticsManager()
        try:
            # setup control log path
            if self.arguments.log_path:
                control_log_path = self.arguments.log_path
                if os.path.exists(control_log_path) and os.path.isdir(control_log_path):
                    routes = self.arguments.routes.split(".")[:-1][0]
                    routes = routes.split("/")[-2:]
                    routes = "_".join(routes)

                    scenarios = self.arguments.scenarios.split(".")[:-1][0]
                    scenarios = scenarios.split("/")[-1:][0]
                    scenarios += "_logs"
                    timestr = time.strftime("%m%d_%H%M%S")
                    if self.arguments.dual_agent:
                        filename = routes + "-" + timestr + "-" + scenarios + "-dual.csv"
                    else:
                        filename = routes + "-" + timestr + "-" + scenarios + "-single.csv"
                    control_log_dir = os.path.join(control_log_path, scenarios)
                    control_log_path = os.path.join(control_log_path, scenarios, filename)
                    if not os.path.isdir(control_log_dir):
                        os.mkdir(control_log_dir)
                else:
                    raise FileNotFoundError("control_log_path needs to be a valid directory")
                self.scene_env = LeaderboardEvaluatorRLEnv(self.arguments, self.statistics_manager,
                                                           control_log_path=control_log_path,
                                                           gym_environment=gym_environment)
            else:
                self.scene_env = LeaderboardEvaluatorRLEnv(self.arguments, self.statistics_manager, gym_environment=gym_environment)
        except Exception as e:
            print(e)
            traceback.print_exc()
            del self.scene_env
            sys.exit(1)

    def run(self):
        print("Running the entire sceanrio.")
        try:
            self.scene_env.run(self.arguments)
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            del self.scene_env

    def prepare_step(self, environment=None):
        print("Prepare for stepping")
        self.scene_env.prepare_step(self.arguments, environment)
        self.prepare_step_flag = True

    def step_sense(self):
        if not self.prepare_step_flag:
            self.prepare_step()
        data, timestamp, rewards = self.scene_env.step_sense()
        return data, timestamp, rewards
    
    def step_act(self, timestamp, action):
        if not self.prepare_step_flag:
            raise RuntimeError("Cannot act before initialization.")
        done = self.scene_env.step_act(timestamp, action)
        return done

    def step_act_vehicle_control(self, timestamp, action, environment):
        if not self.prepare_step_flag:
            raise RuntimeError("Cannot act control before initialization.")
        done = self.scene_env.step_act_control(timestamp, action, environment)
        return done

    def post_step(self):
        print("Post-processing after stepping")
        self.scene_env.after_step(self.arguments)
        self.prepare_step_flag = False
        del self.scene_env

    def reset(self):
        if self.prepare_step_flag:
            self.post_step()
        time.sleep(1)
        self.prepare_step()
