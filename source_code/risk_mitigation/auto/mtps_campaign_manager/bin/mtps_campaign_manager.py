#!/usr/bin/env python
import argparse
import itertools
import os
import json
import shutil
import sys
import glob
from mtps_driver import ProcessRunner
import pickle as pkl
NUM_TRIES = 1


def int_or_float(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def collect_log_files(out_dir, mode):
    print("collecting log files")
    # get diff between the dmesg files and exit
    d_start_file = out_dir + "/dmesg.start"
    d_end_file = out_dir + "/dmesg.end"
    dmesg_start = open(d_start_file).readlines()
    dmesg_end = open(d_end_file).readlines()
    new_line_index = len(dmesg_start)
    dmesg_diff = dmesg_end[new_line_index + 1:]
    with open(out_dir + "/dmesg.err", 'w') as ofh:
        for line in dmesg_diff:
            ofh.write(line)
    if os.path.exists(d_start_file):
        os.remove(d_start_file)
    if os.path.exists(d_end_file):
        os.remove(d_end_file)


class Campaign:
    def __init__(self, name, base_path, scenario, params, script_path):
        self.name = name
        self.base_path = base_path
        self.agent_mode = "SINGLE_AGENT"
        self.scenario = scenario
        self.result_dir = ""
        self.route_name = params["sceneRoute"]
        self.script_path = script_path

    def __str__(self):
        mystr = "name: {} \n" + "num: {} \n" + "result_dir: {} \n" + "enabled: {}"
        return mystr.format(
            self.name,
            str(self.num),
            str(self.enabled)
        )


def launch_epxeriment(exp_mode, cmd, exp_dir, num_run):
    print("\tRunning cmd %s" % cmd)
    print("\tExperiment dir %s" % exp_dir)
    p = ProcessRunner(exp_mode, cmd, ".", "", exp_dir, 420, num_run)
    p.start()
    p.join()
    print("Experiment {} ended.".format(num_run))


def launch_scenario(base_path, scenario, campaign_params, script_path, output_dir, mitigation_mode):
    with open("collisions_rear_ending.pkl", 'rb') as f:
        collision_lists = pkl.load(f)

    campaign = Campaign("{}_parameter_sweeping".format(scenario), base_path, scenario, campaign_params, script_path)
    num_run = 0
    campaign_root_dir = output_dir + os.sep + campaign.agent_mode + "_" + campaign.scenario
    campaign.result_dir = campaign.base_path + os.sep + campaign_root_dir

    # parameter sweeping combinations
    parameterNames = campaign_params["tunableParameters"]["listofNames"]
    combinations = list()
    for r in campaign_params["tunableParameters"]["listofRanges"]:
        r_list = [x for x in range(r[0], r[1], r[2])]
        combinations.append(r_list)
    combinations = list(itertools.product(*combinations))

    # if "ghost_cutin" in campaign_params['sceneName']:
    #     collision_lists = collision_lists[0]
    # elif "lead_cutin" in campaign_params['sceneName']:
    #     collision_lists = collision_lists[1]
    # elif "lead_slowdown" in campaign_params['sceneName']:
    #     collision_lists = collision_lists[2]
    # else:
    #     raise NotImplementedError
    # collision_set = set([int(collision.split('_')[-1]) for collision in collision_lists])

    # if "ghost_cutin" in campaign_params['sceneName']:
    #     collision_set = collision_lists[0]
    # elif "lead_cutin" in campaign_params['sceneName']:
    #     collision_set = collision_lists[1]
    # elif "lead_slowdown" in campaign_params['sceneName']:
    #     collision_set = collision_lists[2]
    # else:
    #     raise NotImplementedError

    collision_set = collision_lists

    # iterate through all combinations and launch each of them
    for combo in combinations:
        # print(num_run, combo)
        # num_run += 1
        # continue
        if num_run in collision_set and num_run > 964:
            # generate configuration dumps
            scenMgrConfigPath = os.path.join(base_path,
                                             campaign_params["basePath2"],
                                             campaign_params["sceneMgrDir"],
                                             "{}.json".format(campaign_params["sceneClassName"]))
            scenConfigDict = dict()
            assert len(combo) == len(parameterNames)
            for idx, _ in enumerate(parameterNames):
                scenConfigDict[parameterNames[idx]] = combo[idx]
            print("Writing scenario configs based on selected parameters:", scenConfigDict)
            with open(scenMgrConfigPath, "w") as f:
                json.dump(scenConfigDict, f)
            result_dir = campaign.result_dir + "_{}".format(str(num_run).zfill(5))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            if not os.path.exists(result_dir + os.sep + "done.txt"):
                files = glob.glob(result_dir + os.sep + "*")
                for f in files:
                    try:
                        os.remove(f)
                    except IsADirectoryError:
                        shutil.rmtree(f)
                shutil.copy(scenMgrConfigPath, result_dir)
                cmd_path = campaign.base_path + os.sep + "mtps_campaign_manager" + os.sep + "bin/"
                cmd = cmd_path + os.sep + "mtps_driver.py --num_run=" + str(
                    num_run) + " --agent_mode " + campaign.agent_mode + " --base_path " + campaign.base_path + " --result_dir " + result_dir + " --timeout " + str(
                    420) + " --script_path " + campaign.script_path + " --scenario_name " + campaign.scenario + " --route_name " + campaign.route_name + " --base_path2 " + campaign_params["basePath2"] + " --mitigation_mode " + mitigation_mode
                launch_epxeriment("param_sweep", cmd, result_dir, num_run)
                ofh = open(result_dir + os.sep + "done.txt", 'w')
                ofh.close()
            os.remove(scenMgrConfigPath)
        num_run += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Scenario config file location")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--mitigation_mode", required=True, help="Mitigation mode")
    params = parser.parse_args()

    # run parameter sweeping
    campaign_parameters = None
    if os.path.exists(params.config):
        with open(params.config, 'r') as fd:
            campaign_parameters = json.load(fd)
    else:
        print("No config file location invalid")
        exit(1)

    base_path = campaign_parameters["basePath"]
    script_path = campaign_parameters["scriptPath"]
    for campaign in campaign_parameters["campaigns"]:
        scenario = campaign["sceneName"]
        launch_scenario(base_path, scenario, campaign, script_path, params.output_dir, params.mitigation_mode)


if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
