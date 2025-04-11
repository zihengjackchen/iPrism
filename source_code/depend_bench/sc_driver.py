#!/usr/bin/env python3

import argparse
import io
import os
import subprocess
import sys
import threading
import time
import signal

import psutil
import re

# agent modes
AGENT_MODES = {
    "SINGLE_AGENT": "SINGLE_AGENT",
    "DUAL_AGENT_DUP": "DUAL_AGENT_DUP",
    "DUAL_AGENT_RR": "DUAL_AGENT_RR",
    "DUAL_AGENT_ASYM": "DUAL_AGENT_ASYM",
    "TRIPLE_AGENT_DUP": "TRIPLE_AGENT_DUP",
    "TRIPLE_AGENT_RR": "TRIPLE_AGENT_RR",
    "TRIPLE_AGENT_ASYM": "TRIPLE_AGENT_ASYM"
}

def get_base_dir(file):
    return os.path.dirname(os.path.realpath(file))


def send_email():
    pass  # TODO


class ProcessRunner(threading.Thread):
    def __init__(self, name, cmd, cwd, driver_mode, result_dir, timeout, num_run):
        threading.Thread.__init__(self)
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.timeout = timeout
        self.result_dir = result_dir
        self.num_run = num_run
        self.mode = driver_mode
        print(self.result_dir)
        if not os.path.exists(self.result_dir):
            print("directory created")
            os.makedirs(self.result_dir)
        self.stdout = self.result_dir + "/" + self.name + ".stdout"
        self.stderr = self.result_dir + "/" + self.name + ".stderr"
        print(self.stdout, self.stderr)
        self.stdout += "." + self.mode
        self.stderr += "." + self.mode

        self.subp = -1  # initialize with an invalid process id
        self.ps = None  # initialize with an invalid process handler
        self.retcode = -1000  # initialize with non standard exit code
        self.writer_stdout = io.open(self.stdout, 'wb')
        self.writer_stderr = io.open(self.stderr, 'wb')
        print("Set timeout at {} seconds.".format(self.timeout))

    def run(self):
        try:
            print("running %s : %s " % (self.name, self.cmd))
            self.subp = subprocess.Popen(self.cmd, shell=True, bufsize=500 * 1024 * 1024,
                                         stdout=self.writer_stdout, stderr=self.writer_stderr, close_fds=True,
                                         cwd=self.cwd)
            self.ps = psutil.Process(self.subp.pid)
        except:
            self.ps = None
            self.subp = None
        self.retcode = -1
        if self.subp is not None and self.ps is not None:
            try:
                ts = self.timeout - (time.time() - self.ps.create_time())
                ts_ = 1 if ts <= 0 else ts
                self.subp.communicate(timeout=ts_)
            except subprocess.TimeoutExpired:
                print(
                    "sending term signal to {} as threshold timeout reached".format(self.name))
                try:
                    # or parent.children() for recursive=False
                    for child in self.ps.children(recursive=True):
                        os.kill(child.pid, signal.SIGINT)
                        time.sleep(5)
                        child.terminate()
                    self.ps.terminate()
                    self.subp.terminate()
                    self.subp.communicate()
                except Exception as e:
                    print(e)
                    pass
            self.retcode = self.subp.returncode


class Driver:
    def __init__(self, base_path, base_path2, script_path, scenario_name, route_name,
                 agent_mode, experiment_mode, num_run, result_dir, device, timeout):
        self.campaign_mode = experiment_mode
        self.agent_mode = agent_mode

        print("campaign_mode is {}".format(self.campaign_mode))
        print("agent_mode is {}".format(self.agent_mode))

        # configure program paths
        self.base_path = base_path
        self.base_path2 = base_path2
        self.script_path = script_path

        # scenario params
        self.scenario_name = scenario_name
        self.route_name = route_name
        self.timeout = timeout

        # define the driver mode, agent, scenario, and fi
        self.driver_mode = agent_mode + "_" + experiment_mode + "_" + self.scenario_name

        # define all kinds of outputs
        self.driver_out_dir = result_dir

        # define which gpu to use
        self.device = device

        # experiment id
        self.num_run = num_run

        # launch time of the program
        self.launch_time = time.time()
        self.programs = {}

        # sim path settings
        self.sim = self.base_path + os.sep + "bin/startsim.sh"

    def cleanup(self, programs):
        for p in programs.keys():
            # force terminate remaining programs
            print("cleaning %s" % p)
            programs[p].timeout = 1
            programs[p].join()

        with open(self.driver_out_dir + os.sep + "exitcodes.txt", "w") as ofh:
            for p in programs:
                ofh.write(programs[p].name + "\t" + programs[p].cmd +
                          "\t" + str(programs[p].retcode) + "\n")

    def mark_failed(self, programs):
        print("Experiment failed :(")
        fd = open(self.driver_out_dir + os.sep + "experiment_status", 'a')
        fd.write("failed\n")
        fd.close()

    def check_experiment_status(self, programs):
        self.mark_success(programs)
        # TODO some checks

    def mark_success(self, programs):
        print("Experiment completed successfully")
        fd = open(self.driver_out_dir + os.sep + "experiment_status", 'w')
        fd.write("success\n")
        fd.close()

    def stop_docker(self):
        print("stopping docker")
        pkill = ProcessRunner("stop_agent", "docker stop dclient", ".",
                              self.driver_mode, self.driver_out_dir, 10, self.num_run)
        pkill.start()
        pkill.join()
        time.sleep(2)

        pkill = ProcessRunner("stop_agent", "docker rm dclient", ".",
                              self.driver_mode, self.driver_out_dir, 10, self.num_run)
        pkill.start()
        pkill.join()

    def stop_sim(self):
        cmd = self.base_path + os.sep + "bin/killsim.sh"
        cwd = self.base_path + os.sep + "bin"
        simkill = ProcessRunner("simkill", cmd, cwd, self.driver_mode,
                                self.driver_out_dir, self.timeout, self.num_run)
        simkill.start()
        simkill.join()
        return

    def run_experiment(self):
        programs = {}
        # run sim ensure that sim is stopped
        self.stop_sim()

        # the launch sim
        sim = ProcessRunner("sim", self.sim, self.driver_out_dir,
                            self.driver_mode, self.driver_out_dir, self.timeout, self.num_run)
        sim.start()
        time.sleep(5)
        print("sim launch done")

        # launch agent, configure agent mode
        out_dir = self.driver_out_dir
        out_dir = os.sep + re.search(self.base_path + ".*", out_dir)[0]
        agent_settings = " --log_path=" + out_dir + " "
        agent_settings += " --sim_data_save=" + out_dir + "/{}_data".format(self.scenario_name) + " "
        agent_settings += " --checkpoint=" + out_dir + os.sep + "results_summary.json" + " "
        agent_settings += " --scenarios={}/{}/leaderboard/data/{}.json".format(self.base_path, self.base_path2,
                                                                               self.scenario_name) + " "
        agent_settings += " --agent=image_agent.py" + " "
        agent_settings += " --routes={}/{}/{}".format(self.base_path, self.base_path2, self.route_name) + " "
        agent_settings += " --port=2000" + " "
        agent_settings += " --trafficManagerSeed=0" + " "
        agent_settings += " --agent-config={}/{}/epoch24.ckpt".format(self.base_path, self.base_path2) + " "
        if self.agent_mode == AGENT_MODES["SINGLE_AGENT"]:
            agent_settings += " "
        else:
            raise ValueError("Unsupported DC agent mode.")

        agent_run_cmd = " python3 " + " " + self.script_path + " " + agent_settings
        print("Experiment mode is {}".format(self.campaign_mode))
        print("Running scenario with hyperparameters, launching agent: {}".format(agent_run_cmd))
        self.programs["agent"] = ProcessRunner( "agent", agent_run_cmd, None, self.driver_mode, self.driver_out_dir, self.timeout, self.num_run)
        self.programs["agent"].start()

        # join all threads before collecting data
        print("wait for the agent to finish or be terminated by the timeout")
        self.programs["agent"].join()
        print("agent process joined")
        time.sleep(5)
        self.stop_sim()
        self.cleanup(self.programs)
        self.check_experiment_status(programs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_mode", default="SINGLE_AGENT",
                        help="mode=single|dual|triple,dup|rr|asym", required=False)
    parser.add_argument("--num_run", default="0",
                        help="experiment number", required=False)
    parser.add_argument("--result_dir", 
                        help="subdirectory to provide at runtime", required=True)
    parser.add_argument("--device", default="1",
                        help="CUDA_VISIBLE_DEVICE", required=False)
    parser.add_argument("--timeout", default=180,
                        help="timeout", required=False, type=int)
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--base_path2", required=True)
    parser.add_argument("--script_path", required=True)
    parser.add_argument("--scenario_name", required=True)
    parser.add_argument("--route_name", required=True)

    params = parser.parse_args()

    dfi = Driver(params.base_path, params.base_path2, params.script_path, params.scenario_name, params.route_name,
                 params.agent_mode, "param_sweep", int(params.num_run), params.result_dir, params.device, params.timeout)
    dfi.run_experiment()


if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
