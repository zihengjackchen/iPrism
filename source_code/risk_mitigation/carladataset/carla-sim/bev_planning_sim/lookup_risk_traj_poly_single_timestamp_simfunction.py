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
import pickle as pkl
import numpy as np
import sys
import copy


class LookupRiskCarlaSimRuntime:
    def __init__(self, risk_pickle_stored, traj_pickle_stored):
        self.risk_pickle_stored = risk_pickle_stored
        with open(self.risk_pickle_stored, "rb") as f:
            self.risk_dict = pkl.load(f)
        print("Loaded risk pickle {}".format(self.risk_pickle_stored))

        self.traj_pickle_stored = traj_pickle_stored
        with open(self.traj_pickle_stored, "rb") as f:
            self.traj_dict = pkl.load(f)
        print("Loaded trajectory pickle {}".format(self.traj_pickle_stored))
        self.ego_telementry_record = self.traj_dict["egoTelemetry"]

    def riskAnalyserSingleTimestamp(self, ego_telemetry):
        # look up risk by finding the closest location
        current_ego_location = ego_telemetry["location"]
        min_dist = sys.float_info.max
        matched_timestamp = -1
        matched_location = None
        for timestamp in self.risk_dict:
            location = self.ego_telementry_record[timestamp]["location"]
            dist = np.linalg.norm(current_ego_location - location)
            if min_dist > dist:
                min_dist = dist
                matched_timestamp = timestamp
                matched_location = copy.deepcopy(location)
        print("Current location:", current_ego_location)
        print("Matched location: {}, timestamp {}".format(matched_location, matched_timestamp))
        return self.risk_dict[matched_timestamp]

