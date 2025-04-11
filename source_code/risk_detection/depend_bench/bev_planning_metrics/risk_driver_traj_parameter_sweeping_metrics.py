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

import glob
import shutil

import path_configs
import subprocess
import os
import sys
import random


if __name__ == "__main__":
    basePath = path_configs.BASEPATH
    subFolders = [str(sys.argv[1])]
    plannerType = str(sys.argv[2])
    suffix = str(sys.argv[3])
    blocking = str(sys.argv[4])
    posUnc = str(sys.argv[5])
    prediction = str(sys.argv[6])
    prefix = None
    if posUnc == "None":
        prefix = "model_unc"
    elif posUnc in ["gaussian2DShift", "gaussian2DRotate", "gaussian2DCorners", "gaussian2DShiftRotate"]:
        if posUnc == "gaussian2DShift":
            prefix = "pos_unc_shift"
        if posUnc == "gaussian2DRotate":
            prefix = "pos_unc_rotate"
        if posUnc == "gaussian2DCorners":
            prefix = "pos_unc_cor"
        if posUnc == "gaussian2DShiftRotate":
            prefix = "pos_unc_shift_rotate"
    else:
        raise Exception("Not supported posUnc")
    if blocking == "nonBlocking":
        raise NotImplementedError
    elif blocking == "blocking":
        for subFolder in subFolders:
            trajDataPath = os.path.join(basePath, subFolder)
            logIDs = sorted([d for d in os.listdir(trajDataPath) if os.path.isdir(os.path.join(trajDataPath, d))])
            # print("!!!! Only run random X folders !!!!")
            # logIDs = random.sample(logIDs, 200)
            # print(logIDs)
            # print("!!!!", logIDs, "!!!!")
            # logIDs = ["SINGLE_AGENT_fi_rear_ending_00000"]
            for logID in logIDs:
                logFile = glob.glob(os.path.join(basePath, subFolder, logID, "**", "route_*.pkl"))
                print(logFile)
                # assert len(logFile) == 1
                logFile = logFile[0].split(logID)[-1]
                logFile = logID + logFile
                new_suffix = suffix + "_{}".format(logID)
                riskSaveDir = os.path.join(basePath, "analysis_risk_{}".format(plannerType), subFolder, logFile, prefix)
                riskSavePkl = os.path.join(riskSaveDir, "{}_risk_analysis.pkl".format(new_suffix))
                riskDestDir = os.path.join(basePath, subFolder, logID)
                riskDestPkl = os.path.join(riskDestDir, "{}_metric_analysis.pkl".format(new_suffix))
                if os.path.isfile(riskDestPkl):
                    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                    print("{} exists, skipping {}.".format(riskDestPkl, logID))
                    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                    continue
                p = subprocess.run(["python3",
                                    basePath + os.sep + "bev_planning_metrics"
                                    "/generate_risk_traj_poly_sts_dispatcher.py",
                                    subFolder, logFile, plannerType, new_suffix, posUnc, prediction])
                print("Move {} to {}.".format(riskSavePkl, riskDestPkl))
                shutil.move(riskSavePkl, riskDestPkl)
    else:
        raise ValueError
    print("Done all logged trajectories.")
