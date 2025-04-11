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
import json
import os
import shutil
import time
import sys
import logging
import subprocess
import path_configs
from generate_bev import GenerateBEVGraph
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.ERROR)
MAX_CONCURRENT_FRAME = int(os.cpu_count())
# MAX_CONCURRENT_FRAME = 1
MAX_CONCURRENT_OBJECTIVES = 1

print("# concurrent frames:", MAX_CONCURRENT_FRAME, "\n# concurrent objectives:", MAX_CONCURRENT_OBJECTIVES)


class GenerateTrajDataArgoverserDispatcher:
    def __init__(self, mapDataPath, trajDataPath, logID, dataSaveDir, scriptPath):
        self.bevGraph = GenerateBEVGraph(mapDataPath=mapDataPath, trajDataPath=trajDataPath, logID=logID)
        self.scriptPath = scriptPath
        self.dataSaveDir = dataSaveDir

    @staticmethod
    def openSubProcess(cmd):
        while True:  # automatically retry
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(">>>>>>> Process:", p.args, "launched >>>>>>>>")
            output, error = p.communicate()
            print("****** Process:", p.args, "finished with return code", p.returncode, "******")
            if p.returncode != 0:
                print("xxxxxx Process:", p.args, "failed with return code", p.returncode, "xxxxxx")
                print("STDOUT content", output)
                print("STDERR content:", error)
                time.sleep(1)
            else:
                print("============ Finish one frame ============")
                break
        return

    def dispatchThenQuit(self, listofArgs, suffix):
        if os.path.isfile(os.path.join(self.dataSaveDir, "data_{}".format(suffix), "done.txt")):
            print("Data present at {}/{}, returning.".format(self.dataSaveDir, suffix))
            return
        cmds = list()
        for timestamp in self.bevGraph.listofTimestamps:
            print("Dispatching timestamp:", timestamp)
            cmd = ["python3", self.scriptPath]
            cmd.extend(listofArgs)
            cmd.extend([str(timestamp), str(MAX_CONCURRENT_OBJECTIVES)])
            cmds.append(cmd)
        with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_FRAME) as executor:
            executor.map(self.openSubProcess, cmds)
        time.sleep(2)
        numInChannels = list()
        for timestamp in self.bevGraph.listofTimestamps:
            saveFileName = "{}_{}_data_generated".format(timestamp, suffix)
            with open(os.path.join(self.dataSaveDir,
                                   "frame_config_{}".format(suffix),
                                   saveFileName + ".json"), "r") as f:
                currentTime_config = json.load(f)
                if len(currentTime_config):
                    numInChannels.append(int(currentTime_config["numInChannels"]))
        print(numInChannels)
        assert all(x == numInChannels[0] for x in numInChannels)

        saveFileName = "{}_{}_data_generated".format(self.bevGraph.listofTimestamps[0], suffix)
        consolidateSaveFileName = "{}_data_config".format(suffix)
        src = os.path.join(self.dataSaveDir, "frame_config_{}".format(suffix), saveFileName + ".json")
        dest = os.path.join(self.dataSaveDir, consolidateSaveFileName + ".json")
        shutil.copy(src, dest)
        shutil.move(os.path.join(self.dataSaveDir, "frame_config_{}".format(suffix)),
                    os.path.join(self.dataSaveDir, "data_{}".format(suffix)))
        shutil.move(dest, os.path.join(self.dataSaveDir, "data_{}".format(suffix), consolidateSaveFileName + ".json"))
        print("Check passed.")
        time.sleep(2)
        f = open(os.path.join(self.dataSaveDir, "data_{}".format(suffix), "done.txt"), "w")
        f.close()
        print("Finish generating per frame datapoints.")


if __name__ == "__main__":
    basePath = path_configs.BASEPATH
    mapDataPath = os.path.join(basePath, "map_files")
    subFolder = sys.argv[1]
    logID = sys.argv[2]
    plannerType = sys.argv[3]
    assert plannerType == "dyn", "Planner type for trajectory based risk analysis can only be dyn"
    suffix = sys.argv[4]
    posUnc = sys.argv[5]
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
    trajDataPath = os.path.join(basePath, "argoverse-tracking", subFolder)
    routeVisDir = os.path.join(basePath, "visualize_routing_{}".format(plannerType), subFolder, logID, prefix, suffix)
    riskVisDir = os.path.join(basePath, "visualize_risk_{}".format(plannerType), subFolder, logID, prefix, suffix)
    riskSaveDir = os.path.join(basePath, "analysis_risk_{}".format(plannerType), subFolder, logID, prefix)
    dataSaveDir = os.path.join(basePath, "generate_data_{}".format(plannerType), subFolder, logID, prefix)
    dataVisDir = os.path.join(basePath, "visualize_data_{}".format(plannerType), subFolder, logID, prefix, suffix)
    if not os.path.exists(routeVisDir):
        os.makedirs(routeVisDir)
    if not os.path.exists(riskVisDir):
        os.makedirs(riskVisDir)
    if not os.path.exists(dataVisDir):
        os.makedirs(dataVisDir)
    if not os.path.exists(dataSaveDir):
        try:
            os.makedirs(dataSaveDir)
        except FileExistsError:
            print("Folder exist continue...")
    if not os.path.exists(os.path.join(dataSaveDir, "data_{}".format(suffix))):
        try:
            os.makedirs(os.path.join(dataSaveDir, "data_{}".format(suffix)))
        except FileExistsError:
            print("Temp folder exist continue...")
    if not os.path.exists(os.path.join(dataSaveDir, "frame_config_{}".format(suffix))):
        try:
            os.makedirs(os.path.join(dataSaveDir, "frame_config_{}".format(suffix)))
        except FileExistsError:
            print("Temp folder exist continue...")
    genDataDisp = GenerateTrajDataArgoverserDispatcher(
        mapDataPath=mapDataPath,
        trajDataPath=trajDataPath,
        logID=logID,
        dataSaveDir=dataSaveDir,
        scriptPath=os.path.join(basePath, "bev_planning_pkl",
                                "generate_traj_data_poly_single_timestamp.py"))
    listofArgs = [subFolder, logID, plannerType, suffix, posUnc]
    if len(sys.argv) > 6:
        seed = sys.argv[6]
        listofArgs.append(seed)
    print("List of arguements:", listofArgs)
    genDataDisp.dispatchThenQuit(listofArgs, suffix)
