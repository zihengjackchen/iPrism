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

import traceback
import argparse
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
import carla_sim_datautils

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


class LeaderboardEvaluator(object):
    """
    TODO: document me!
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 40.0  # in Hz

    def __init__(self, args, statistics_manager, control_log_path=None):
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
        self.client = carla.Client(args.host, int(args.port))
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        dist = pkg_resources.get_distribution("carla")
        if dist.version != 'leaderboard':
            if LooseVersion(dist.version) < LooseVersion('0.9.10'):
                raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, args.debug > 1, control_log_path=control_log_path)

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

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """

        self.world = self.client.load_world(town)
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

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

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

            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            sys.exit(-1)

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("Could not set up the required agent:")
            print("> {}\n".format(e))
            traceback.print_exc()

            crash_message = "Agent couldn't be set up"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            return

        print("Loading the world")

        # Load the world and the scenario
        try:
            # load the world
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)

            # collect all static information
            routes = args.routes.split("/")[-1]
            scenarios = args.scenarios.split("/")[-1]
            dc = carla_sim_datautils.DataCollector(args.sim_data_save, routes + "_" + scenarios+".pkl",
                                                    self.world, self.world.get_map())
            dc.collect_lane_information(config, travel_dist=4.5)
            dc.collect_static_obstacles()
            self.manager.set_data_collector(dc)

            self._prepare_ego_vehicles(config.ego_vehicles, False)

            # change to fault injection scenario
            if "fi" in args.routes:
                scenario = FIScenario(world=self.world, config=config, debug_mode=args.debug)
            else:
                scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)

            self.statistics_manager.set_scenario(scenario.scenario)

            # Night mode
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))

            self.manager.load_scenario(scenario,
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

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()
            sys.exit(-1)

        print("Running the route")

        # Setup Fault Injector check if flag is set and the layer is valid
        print("Configuring pyTorchFI")
        if args.enable_fi and args.layer != -1:
            pfi_model = self.agent_instance.get_pfi_model()
            pfi_carla_utils.print_pfi_model(pfi_model)
            pfi_inj = None
            layer = args.layer
            k = args.k
            c = args.c
            h = args.h
            w = args.w
            minval = args.minval
            maxval = args.maxval
            layer_name = args.layer_name
            usemask = args.usemask
            if args.fi_mode == "perm":
                if args.fi_type == "weight":
                    pfi_inj = pfi_carla_utils.single_weight_injection(pfi_model, layer, k, c, h, w, minval,
                                                                      maxval, layer_name, usemask)
                elif args.fi_type == "neuron":
                    pfi_inj = pfi_carla_utils.single_neuron_injection(pfi_model, layer, c, h, w, minval,
                                                                      maxval, layer_name, usemask)
                else:
                    raise ValueError("Unknown injection type:", args.fi_type)
            elif args.fi_mode == "tran":
                raise NotImplementedError("Tranisient not implemented yet...")
            self.agent_instance.set_pfi_inj(pfi_inj)
        else:
            print("Fault Injection Disabled.")

        # Run the scenario
        try:
            self.manager.run_scenario()

        except AgentError as e:
            # The agent has failed -> stop the route
            print("\nStopping the route, the agent has crashed:")
            print("> {}\n".format(e))
            traceback.print_exc()

            crash_message = "Agent crashed"

        except Exception as e:
            print("\nError during the simulation:")
            print("> {}\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

        # Stop the scenario
        try:
            print("Stopping the route")
            self.manager.stop_scenario()
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            # save data collection results
            dc.save_collected_data()

            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            scenario.remove_all_actors()

            self._cleanup()

        except Exception as e:
            print("\nFailed to stop the scenario, the statistics might be empty:")
            print("> {}\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"

        if crash_message == "Simulation crashed":
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

            # run
            self._load_and_run_scenario(args, config)

            route_indexer.save_state(args.checkpoint)

        # save global statistics
        print("Registering the global statistics")
        global_stats_record = self.statistics_manager.compute_global_statistics(route_indexer.total)
        StatisticsManager.save_global_record(global_stats_record, self.sensor_icons, route_indexer.total,
                                             args.checkpoint)


def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='8000',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="120.0",
                        help='Set the CARLA client timeout value in seconds')
    parser.add_argument('--sim_data_save', required=True, help="place to save sim data")

    # simulation setup
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        required=True)
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        required=True)
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    # diverse AV project additional arugments
    parser.add_argument("--dual_agent", action="store_true")
    parser.add_argument("--dual_dup", action="store_true",
                        help="with the flag we duplicate the agent, without this flag we do round robin")
    parser.add_argument("--log_path", default=None, type=str)

    # enable pytorchfi for weight and neuron injection
    parser.add_argument("--enable_fi", action="store_true",
                        help="enable pytorchfi, must also input mode, and position of injection")
    parser.add_argument("--fi_type", type=str, help="mode of fi, valid mode: weight, neuron", default="")
    parser.add_argument("--fi_mode", type=str, help="type of fi, valid type: perm, tran", default="")
    parser.add_argument("--layer_name", type=str, help="name of the injection layer", default="")
    parser.add_argument("--layer", type=int, help="targeted conv2d layer")
    parser.add_argument("--k", type=int, help="targeted conv2d feature map, ignore wihen type=neuron", default=None)
    parser.add_argument("--c", type=int, help="targeted conv2d channel", default=None)
    parser.add_argument("--h", type=int, help="targeted conv2d height", default=None)
    parser.add_argument("--w", type=int, help="targeted conv2d width", default=None)
    parser.add_argument("--minval", type=int, help="targeted conv2d height", default=None)
    parser.add_argument("--maxval", type=int, help="targeted conv2d width", default=None)
    parser.add_argument("--usemask", action="store_true", help="use the mask instead of directly change the value")
    parser.add_argument("--preprocessing", type=str, default=None,
                        help="use the mask instead of directly change the value")
    parser.add_argument("--preparam", default=[], nargs='+', help="use the mask instead of directly change the value",
                        type=float)

    arguments = parser.parse_args()

    statistics_manager = StatisticsManager()

    if arguments.enable_fi:
        print("========================= Warning ! =========================")
        print("To use weight based pytorch FI injection, make sure you have")
        print("replaced the pytorch fi core.py file in the .local python lib")
        print("with the one provided by the repository.")
        print("=========================+++++++++++=========================")
        if not arguments.fi_mode or not arguments.fi_type:
            print("Must also set fi_mode, fi_type, and position variable for pytorchfi, exiting...")
            return

    try:
        # setup control log path
        if arguments.log_path:
            control_log_path = arguments.log_path
            if os.path.exists(control_log_path) and os.path.isdir(control_log_path):
                routes = arguments.routes.split(".")[:-1][0]
                routes = routes.split("/")[-2:]
                routes = "_".join(routes)

                scenarios = arguments.scenarios.split(".")[:-1][0]
                scenarios = scenarios.split("/")[-1:][0]
                scenarios += "_logs"
                timestr = time.strftime("%m%d_%H%M%S")
                if arguments.dual_agent:
                    filename = routes + "-" + timestr + "-" + scenarios + "-dual.csv"
                else:
                    filename = routes + "-" + timestr + "-" + scenarios + "-single.csv"
                control_log_dir = os.path.join(control_log_path, scenarios)
                control_log_path = os.path.join(control_log_path, scenarios, filename)
                if not os.path.isdir(control_log_dir):
                    os.mkdir(control_log_dir)
            else:
                raise FileNotFoundError("control_log_path needs to be a valid directory")
            leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager,
                                                         control_log_path=control_log_path)
        else:
            leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
        leaderboard_evaluator.run(arguments)

    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == '__main__':
    main()
