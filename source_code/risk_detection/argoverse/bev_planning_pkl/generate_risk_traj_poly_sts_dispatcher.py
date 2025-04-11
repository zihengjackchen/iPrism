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

import os
import shutil
import time
import sys
import logging
import pickle as pkl
import subprocess
import path_configs
from generate_bev import GenerateBEVGraph
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.ERROR)
# MAX_CONCURRENT_FRAME = int(os.cpu_count() / 1)
MAX_CONCURRENT_FRAME = 5
# MAX_CONCURRENT_FRAME = 1
MAX_CONCURRENT_OBJECTIVES = 1

print("# concurrent frames:", MAX_CONCURRENT_FRAME, "\n# concurrent objectives:", MAX_CONCURRENT_OBJECTIVES)


class GenerateRiskArgoverserDispatcher:
    def __init__(self, mapDataPath, trajDataPath, logID, riskSaveDir, scriptPath):
        self.bevGraph = GenerateBEVGraph(mapDataPath=mapDataPath, trajDataPath=trajDataPath, logID=logID)
        self.scriptPath = scriptPath
        self.riskSaveDir = riskSaveDir

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
                print("STDOUT content", output.decode('utf-8'))
                print("============ Finish one frame ============")
                break
        return

    def dispatchThenReduce(self, listofArgs, suffix):
        saveFileNameFinal = "{}_risk_analysis.pkl".format(suffix)
        if os.path.isfile(os.path.join(self.riskSaveDir, saveFileNameFinal)):
            print("{} exists, already analysed, return...".format(os.path.join(self.riskSaveDir, saveFileNameFinal)))
            shutil.rmtree(os.path.join(self.riskSaveDir, "tmp_{}".format(suffix)))
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
        aggregateResults = dict()
        for timestamp in self.bevGraph.listofTimestamps:
            saveFileName = "{}_risk_analysis.pkl.{}".format(suffix, timestamp)
            print("Aggragating:", saveFileName)
            with open(os.path.join(self.riskSaveDir, "tmp_{}".format(suffix), saveFileName), "rb") as f:
                pkl_dump = pkl.load(f)
                aggregateResults[timestamp] = pkl_dump[timestamp]

        with open(os.path.join(self.riskSaveDir, saveFileNameFinal), "wb") as f:
            pkl.dump(aggregateResults, f)
        time.sleep(2)
        print("Saved analysis pkl file to:", os.path.join(self.riskSaveDir, saveFileNameFinal))
        print("Cleanup tmp dir:", os.path.join(self.riskSaveDir, "tmp_{}".format(suffix)))
        shutil.rmtree(os.path.join(self.riskSaveDir, "tmp_{}".format(suffix)))


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
    riskVisDir = os.path.join(basePath, "visualize_risk_RealEnv_{}".format(plannerType), subFolder, logID, prefix, suffix)
    riskSaveDir = os.path.join(basePath, "analysis_risk_RealEnv_{}".format(plannerType), subFolder, logID, prefix)
    if not os.path.exists(routeVisDir):
        os.makedirs(routeVisDir)
    if not os.path.exists(riskVisDir):
        os.makedirs(riskVisDir)
    if not os.path.exists(riskSaveDir):
        try:
            os.makedirs(riskSaveDir)
        except FileExistsError:
            print("Folder exist continue...")
    if not os.path.exists(os.path.join(riskSaveDir, "tmp_{}".format(suffix))):
        try:
            os.makedirs(os.path.join(riskSaveDir, "tmp_{}".format(suffix)))
        except FileExistsError:
            print("Temp folder exist continue...")
    genRiskDisp = GenerateRiskArgoverserDispatcher(mapDataPath=mapDataPath,
                                                   trajDataPath=trajDataPath,
                                                   logID=logID,
                                                   riskSaveDir=riskSaveDir,
                                                   scriptPath=os.path.join(basePath, "bev_planning_pkl",
                                                                           "generate_risk_traj_poly_single_timestamp.py"))
    listofArgs = [subFolder, logID, plannerType, suffix, posUnc]
    if len(sys.argv) > 6:
        seed = sys.argv[6]
        listofArgs.append(seed)
    print("List of arguements:", listofArgs)
    genRiskDisp.dispatchThenReduce(listofArgs, suffix)
