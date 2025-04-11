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
import os
import sys
import random

if __name__ == "__main__":
    basePath = path_configs.BASEPATH
    mapDataPath = os.path.join(basePath, "map_files")
    # subFolders = ["sample", "train1", "train2", "train3", "train4", "val"]
    subFolders = [str(sys.argv[1])]
    plannerType = str(sys.argv[2])
    suffix = str(sys.argv[3])
    blocking = str(sys.argv[4])
    posUnc = str(sys.argv[5])
    if blocking == "nonBlocking":
        processList = list()
        for subFolder in subFolders:
            trajDataPath = os.path.join(basePath, "argoverse-tracking", subFolder)
            logIDs = [d for d in os.listdir(trajDataPath) if os.path.isdir(os.path.join(trajDataPath, d))]
            random.shuffle(logIDs)
            print(logIDs)
            for logID in logIDs:
                p = subprocess.Popen(["python3",
                                      "{}/bev_planning".format(path_configs.BASEPATH)
                                      "/generate_risk.py",
                                      subFolder, logID, plannerType, suffix, posUnc])
                processList.append(p)
        print("Process list:", processList)
        exit_codes = [p.wait() for p in processList]
    elif blocking == "blocking":
        for subFolder in subFolders:
            trajDataPath = os.path.join(basePath, "argoverse-tracking", subFolder)
            logIDs = [d for d in os.listdir(trajDataPath) if os.path.isdir(os.path.join(trajDataPath, d))]
            random.shuffle(logIDs)
            print(logIDs)
            for logID in logIDs:
                p = subprocess.run(["python3",
                                    "{}/bev_planning".format(path_configs.BASEPATH)
                                    "/generate_risk.py",
                                    subFolder, logID, plannerType, suffix, posUnc])
    else:
        raise ValueError
    print("Done all logged trajectories.")
