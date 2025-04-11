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
            # logIDs = random.sample(logIDs, 1)
            # print(logIDs)
            # print("!!!!", logIDs, "!!!!")
            # logIDs = ["SINGLE_AGENT_fi_ghost_cutin_00216"]
            for logID in logIDs:
                logFile = glob.glob(os.path.join(basePath, subFolder, logID, "**", "*.pkl"))
                assert len(logFile) == 1
                logFile = logFile[0].split(logID)[-1]
                logFile = logID + logFile
                new_suffix = suffix + "_{}".format(logID)
                data_zip_dir = os.path.join(
                    basePath, "generate_data_{}".format(plannerType), subFolder, logFile, prefix, "data_{}".format(new_suffix)
                )
                data_zip_file = os.path.join(
                    basePath, "generate_data_{}".format(plannerType), subFolder, logFile, prefix, "data_{}.zip".format(new_suffix)
                )
                if os.path.isfile(data_zip_file):
                    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                    print("{} exists, data already generated, skipping {}.".format(data_zip_file, logID))
                    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                    continue
                p = subprocess.run(["python3",
                                    basePath + os.sep + "bev_planning"
                                    "/generate_traj_data_poly_sts_dispatcher.py",
                                    subFolder, logFile, plannerType, new_suffix, posUnc, prediction])
                print("Compressing generated data {}.".format(data_zip_dir))
                shutil.make_archive(data_zip_dir, 'zip', data_zip_dir)
                shutil.rmtree(data_zip_dir)
    else:
        raise ValueError
    print("Done all logged trajectories.")
