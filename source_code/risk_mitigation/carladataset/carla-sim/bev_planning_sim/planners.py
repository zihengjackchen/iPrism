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
import matplotlib.pyplot as plt
import numpy as np
import rrt_star_wrapper
import hybrid_astar_wrapper
# import PythonRobotics.PathPlanning.FrenetOptimalTrajectory.frenet_wrapper_parallel as frenet_wrapper
import PythonRobotics.PathPlanning.FrenetOptimalTrajectory.frenet_wrapper as frenet_wrapper
import time


class RRTStarPlanner:
    def __init__(self):
        self.hyperParameters = {
            "step_size": 1.0,
            "max_iterations": 5000,
            "end_dist_threshold": 0.5,
            "obstacle_clearance": 0.3,
            "lane_width": 30.0,
        }

    def inference(self, initialConditions, seed):
        return rrt_star_wrapper.apply_rrt_star(initialConditions, self.hyperParameters, seed)


class HybridAStarPlanner:
    def __init__(self):
        self.hyperParameters = {
            "step_size": 1,
            "max_iterations": 500,
            "completion_threshold": 0.5,
            "angle_completion_threshold": 1.5,
            "rad_step": 0.75,
            "rad_upper_range": 2.0,
            "rad_lower_range": 2.0,
            "obstacle_clearance": 0.1,
            "lane_width": 30.0,
            "radius": 6.0,
            "car_length": 1.0,
            "car_width": 1.0,
        }

    def inference(self, initialConditions):
        return hybrid_astar_wrapper.apply_hybrid_astar(initialConditions, self.hyperParameters)


class FOTPlanner:
    @staticmethod
    def inference(initial_state, reference_waypoints, time_based_obstacles, hyper_parameters):
        return frenet_wrapper.apply_fot(initial_state, reference_waypoints, time_based_obstacles, hyper_parameters)
