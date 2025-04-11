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

from concurrent.futures import ProcessPoolExecutor
import subprocess
import os
import sys
import random
import path_configs
import glob
import shutil


def dispatcher(arguements):
    print("In dispatcher")
    subFolder = arguements["subFolder"]
    logID = arguements["logID"]
    plannerType = arguements["plannerType"]
    suffix = arguements["suffix"]
    posUnc = arguements["posUnc"]
    p = subprocess.run(
        ["python3",
         "{}/bev_planning_pkl/generate_traj_data_poly_sts_dispatcher.py".format(path_configs.BASEPATH),
         subFolder, logID, plannerType, suffix, posUnc
         ]
    )
    print(p, "Finished")


if __name__ == "__main__":
    basePath = path_configs.BASEPATH
    mapDataPath = os.path.join(basePath, "map_files")
    subFolders = str(sys.argv[1]).split(',')
    plannerType = str(sys.argv[2])
    suffix = str(sys.argv[3])
    blocking = str(sys.argv[4])
    posUnc = str(sys.argv[5])

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

    if blocking == "nonBlocking":
        argumentsList = list()
        for subFolder in subFolders:
            trajDataPath = os.path.join(basePath, "argoverse-tracking", subFolder)
            logIDs = [d for d in os.listdir(trajDataPath) if os.path.isdir(os.path.join(trajDataPath, d))]
            random.shuffle(logIDs)
            print(logIDs)
            for logID in logIDs:
                argumentsList.append(
                    {"subFolder": subFolder, "logID": logID, "plannerType": plannerType, "suffix": suffix,
                     "posUnc": posUnc})
        with ProcessPoolExecutor() as executor:
            executor.map(dispatcher, argumentsList)

    elif blocking == "blocking":
        for subFolder in subFolders:
            trajDataPath = os.path.join(basePath, "argoverse-tracking", subFolder)
            logIDs = [d for d in os.listdir(trajDataPath) if os.path.isdir(os.path.join(trajDataPath, d))]
            random.shuffle(logIDs)
            print(logIDs)
            for logID in logIDs:

                # logFile = glob.glob(os.path.join(basePath, subFolder, logID, "**", "*.pkl"))
                # logFile = logFile[0].split(logID)[-1]
                # logFile = logID + logFile
                # new_suffix =  + "_{}".format()
                data_zip_dir = os.path.join(
                    basePath, "generate_data_{}".format(plannerType), subFolder, logID, prefix,
                    "data_{}".format(suffix)
                )
                data_zip_file = os.path.join(
                    basePath, "generate_data_{}".format(plannerType), subFolder, logID, prefix,
                    "data_{}.zip".format(suffix)
                )
                if os.path.isfile(data_zip_file):
                    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                    print("{} exists, data already generated, skipping {}.".format(data_zip_file, logID))
                    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                    continue

                p = subprocess.run(["python3",
                                    "{}/bev_planning_pkl/generate_traj_data_poly_sts_dispatcher.py".format(path_configs.BASEPATH),
                                    subFolder, logID, plannerType, suffix, posUnc])

                print("Compressing generated data {}.".format(data_zip_dir))
                shutil.make_archive(data_zip_dir, 'zip', data_zip_dir)
                shutil.rmtree(data_zip_dir)
    else:
        raise ValueError
    print("Done all logged trajectories.")
