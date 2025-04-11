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


import subprocess
import sys
import path_configs

if __name__ == "__main__":
    campaignType = sys.argv[1]
    subFolder = sys.argv[2]
    lowerRunRange = int(sys.argv[3])
    upperRunRange = int(sys.argv[4])
    plannerType = sys.argv[5]
    prefix = sys.argv[6]
    blocking = sys.argv[7]
    if campaignType == "modelUnc":
        assert upperRunRange > lowerRunRange
        if blocking == "blocking":
            for r in range(lowerRunRange, upperRunRange, 1):
                print("============== run {} ==============".format(r))
                subprocess.run(["python3",
                                "{}/bev_planning/risk_driver_traj_queued.py".format(path_configs.BASEPATH),
                                subFolder, plannerType, prefix+"_"+str(r), "blocking", "None"])
                print("************** run {} **************".format(r))
        elif blocking == "nonBlocking":
            processList = list()
            for r in range(lowerRunRange, upperRunRange, 1):
                print("============== run {} ==============".format(r))
                p = subprocess.Popen(["python3",
                                      "{}/bev_planning/risk_driver_traj_queued.py".format(path_configs.BASEPATH),
                                      subFolder, plannerType, prefix+"_"+str(r), "blocking", "None"])
                print("************** run {} **************".format(r))
                processList.append(p)
            print("Process list:", processList)
            exit_codes = [p.wait() for p in processList]
            print("Done processing all processes.")
    if campaignType == "positionUnc":
        assert upperRunRange > lowerRunRange
        posUnc = sys.argv[8]
        if blocking == "blocking":
            for r in range(lowerRunRange, upperRunRange, 1):
                print("============== run {} ==============".format(r))
                subprocess.run(["python3",
                                "{}/bev_planning/risk_driver_traj_queued.py".format(path_configs.BASEPATH),
                                subFolder, plannerType, prefix+"_"+str(r), "blocking", posUnc])
                print("************** run {} **************".format(r))
        elif blocking == "nonBlocking":
            processList = list()
            for r in range(lowerRunRange, upperRunRange, 1):
                print("============== run {} ==============".format(r))
                p = subprocess.Popen(["python3",
                                      "{}/bev_planning/risk_driver_traj_queued.py".format(path_configs.BASEPATH),
                                      subFolder, plannerType, prefix+"_"+str(r), "blocking", posUnc])
                print("************** run {} **************".format(r))
                processList.append(p)
            print("Process list:", processList)
            exit_codes = [p.wait() for p in processList]
            print("Done processing all processes.")
