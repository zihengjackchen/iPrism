import argparse
import torch
from yaml import parse
from mitigation_rl_dqn import CarlaLBCDQN
# from mitigation_rl_rip_sti import CarlaRIP


def main():
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--sim_data_save", required=True)
    parser.add_argument("--mitigation_risk_save", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--agent", required=True)

    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--trafficManagerSeed", required=True, type=int)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--mitigation_mode", required=True)

    params = parser.parse_args()

    dqn_lbc = CarlaLBCDQN()
    dqn_lbc.config_dict["log_path"] = params.log_path
    dqn_lbc.config_dict["sim_data_save"] = params.sim_data_save
    dqn_lbc.config_dict["checkpoint"] = params.checkpoint
    dqn_lbc.config_dict["agent"] = params.agent
    dqn_lbc.config_dict["scenarios"] = params.scenarios
    dqn_lbc.config_dict["routes"] = params.routes
    dqn_lbc.config_dict["port"] = params.port
    dqn_lbc.config_dict["agent-config"] = vars(params)["agent_config"]
    dqn_lbc.config_dict["mitigation_risk_save"] = params.mitigation_risk_save
    dqn_lbc.config_dict["log_path"] = params.log_path
    dqn_lbc.inference_loop_standalone_process(total_episodes=1, mitigation=params.mitigation_mode, mitigation_configs={
        "skip_mitigation_seconds": 5,
        "skip_first_seconds": 4,
        "constant_threshold": 0.4,
    })
    #
    # dqn_rip = CarlaRIP()
    # dqn_rip.config_dict["log_path"] = params.log_path
    # dqn_rip.config_dict["sim_data_save"] = params.sim_data_save
    # dqn_rip.config_dict["checkpoint"] = params.checkpoint
    # dqn_rip.config_dict["agent"] = params.agent
    # dqn_rip.config_dict["scenarios"] = params.scenarios
    # dqn_rip.config_dict["routes"] = params.routes
    # dqn_rip.config_dict["port"] = params.port
    # dqn_rip.config_dict["agent-config"] = vars(params)["agent_config"]
    # dqn_rip.config_dict["mitigation_risk_save"] = params.mitigation_risk_save
    # dqn_rip.config_dict["log_path"] = params.log_path
    # dqn_rip.inference_loop_standalone_process(total_episodes=1, mitigation=params.mitigation_mode, mitigation_configs={
    #     "skip_mitigation_seconds": 5,
    #     "skip_first_seconds": 4,
    #     "constant_threshold": 0.4,
    # })


if __name__ == "__main__":
    main()
