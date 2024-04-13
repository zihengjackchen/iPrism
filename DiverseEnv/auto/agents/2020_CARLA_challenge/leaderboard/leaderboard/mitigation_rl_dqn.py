import pickle
import os.path
import time
import math
import copy
import random
import subprocess
from torch import nn
import torch.optim as optim
import torch
import torchvision
import matplotlib.pyplot as plt
from carla_project.src.dqn_models import QvalueModel
from carla_project.src.dqn_models import ActorModel
from leaderboard.env_interactor import CarlaRLEnv
from leaderboard.replay_buffer import ReplayBuffer
import numpy as np
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Process, Manager
from torchvision import transforms

writer = SummaryWriter()
PLOT = True


class CarlaLBCDQN(object):
    def __init__(self):

        self.config_dict = {
            'host': 'localhost',
            'port': '2000',
            'trafficManagerPort': 8000,
            'trafficManagerSeed': 0,
            'debug': 0,
            'record': '',
            'timeout': 2500,
            'sim_data_save': '/<PATH_TO_FILE>/DSN2024/DiverseEnv/auto/sim_data_collection',
            # 'routes': '/<PATH_TO_FILE>/DSN2024/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard/data/routes_fi/route_highway.xml',
            # 'scenarios': '/<PATH_TO_FILE>/DSN2024/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard/data/fi_ghost_cutin.json',

            # 'routes': '/<PATH_TO_FILE>/DSN2024/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard/data/routes_fi/route_highway_curved.xml',
            # 'scenarios': '/<PATH_TO_FILE>/DSN2024/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard/data/fi_ghost_cutin_curved.json',

            'routes': '/<PATH_TO_FILE>/DSN2024/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard/data/routes_fi/route_highway_curved.xml',
            'scenarios': '/<PATH_TO_FILE>/DSN2024/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard/data/fi_lead_cutin_curved.json',

            'repetitions': 1,
            'agent': 'image_agent_rl.py',
            'agent_config': '/<PATH_TO_FILE>/DSN2024/DiverseEnv/auto/agents/2020_CARLA_challenge/epoch24.ckpt',
            'track': 'SENSORS',
            'resume': False,
            'checkpoint': './simulation_results.json',
            'dual_agent': False,
            'dual_dup': False,
            'log_path': None,
            'enable_fi': False,
            'fi_type': '',
            'fi_mode': '',
            'layer_name': '',
            'layer': None,
            'k': None,
            'c': None,
            'h': None,
            'w': None,
            'minval': None,
            'maxval': None,
            'usemask': False,
            'preprocessing': None,
            'preparam': [],
            'risk_evaluation_mode': "dyn",
            'risk_pickle_stored': "/<PATH_TO_FILE>/DSN2024/DiverseEnv/auto/sim_risk_analyzed/risk_lookup_3_risk_analysis.pkl",
            'traj_pickle_stored': "/<PATH_TO_FILE>/DSN2024/DiverseEnv/carladataset/carla-sim/safety-critical-simdata/route_highway.xml_fi_ghost_cutin.json.pkl",
            'inference_model_path': "/<PATH_TO_FILE>/projects/ReachML/reachml/checkpoints/final_lead_ghost_cutin_cvctr_3s_train_2022-05-15-16-17-17.pth",
            'inference_config_path': "/<PATH_TO_FILE>/projects/ReachML/reachml/data/train_test_sim_lite/route_highway.xml_fi_ghost_cutin.json.pkl/model_unc/data_realtime_trail0_3s/realtime_trail0_3s_data_config.json",
            'qvalue_net_rl': 1e-5,
            'weight_decay': 0.9999,
            'reward_decay': 0.99,
            'tau': 0.01,
            "rl_checkpoint_savedir": "/<PATH_TO_FILE>/DSN2024/DiverseEnv/auto/rl_checkpoint",
            'replay_buffer_size': 5000,
            'action_space': 2,
            'base_path': '/<PATH_TO_FILE>/projects/auto',
            # read from inference
            # 'ram_shared_path': '/<PATH_TO_FILE>/data9/<PATH_TO_FILE>/DiverseAV/auto/ram_shared/dqn_sti_online/'
            'ram_shared_path': '/<PATH_TO_FILE>/DSN2024/DiverseEnv/auto/ram_shared/dqn_sti_online'
        }

        self.carla_rl_env = None
        self.checkpoint_dict = torch.load(self.config_dict["agent_config"])
        self.hparams = self.checkpoint_dict["hparams"]

        # define action space
        self.hparams["action_space"] = self.config_dict["action_space"]

        # insert LBC actor net, evalution only, this is part of the "environment"
        self.policy_net = ActorModel(self.hparams)
        self.policy_net.load_state_dict(self.checkpoint_dict["state_dict"], strict=False)
        self.policy_net.cuda()

        # insert LBC-based Qvalue net and target net
        self.qvalue_net = QvalueModel(self.hparams)
        self.qvalue_net.cuda()
        self.qvalue_net_target = QvalueModel(self.hparams)
        for target_param, param in zip(self.qvalue_net_target.parameters(), self.qvalue_net.parameters()):
            target_param.data.copy_(param.data)
        self.qvalue_net_target.cuda()

        # insert replay buffer
        self.replay_buffer = ReplayBuffer(self.config_dict["replay_buffer_size"])

        # insert critic_criterion and optimizers
        self.qvalue_net_criterion = nn.MSELoss(reduction='sum')
        self.qvalue_net_optimizer = optim.Adam(self.qvalue_net.parameters(),
                                               lr=self.config_dict["qvalue_net_rl"],
                                               weight_decay=self.config_dict["weight_decay"])

        # define action encoding
        self.ACTION_MAP = {
            0: "NOOP",
            1: "EB",
            2: "ML",
            3: "MR"
        }

        # image processing
        self.grayscale_trans = transforms.Grayscale()

        # bookkeeping information
        self.total_frame_counter = 0
        self.ram_shared_path = self.config_dict["ram_shared_path"]
        self.total_update_steps = 0
        self.all_rewards = []
        self.avg_rewards = []
        self.repeated_noop = False
        self.last_action = -1
        self.replay_buffer_cleared = False
        self.second_action = False
        self.previous_action = 0
        self.first_run = True

        # simulator start and stop
        self.sim_start = os.path.join(self.config_dict["base_path"], os.path.join("bin", "startsim.sh"))
        self.sim_kill = os.path.join(self.config_dict["base_path"], os.path.join("bin", "killsim.sh"))

    def inference_loop_standalone_process(self, total_episodes, mitigation, mitigation_configs):
        try:
            for i in range(total_episodes):
                self.episode_reward = 0
                self.start_sim()
                if mitigation == "none":
                    skip_mitigation_seconds = mitigation_configs["skip_mitigation_seconds"]
                    skip_first_seconds = mitigation_configs["skip_first_seconds"]
                    p = Process(target=self.inference_single_episode_rl_mitigation, args=(i, skip_first_seconds, skip_mitigation_seconds, False))
                elif mitigation == "constant":
                    skip_mitigation_seconds = mitigation_configs["skip_mitigation_seconds"]
                    skip_first_seconds = mitigation_configs["skip_first_seconds"]
                    const_threshold = mitigation_configs["constant_threshold"]
                    p = Process(target=self.inference_single_episode_rl_mitigation, args=(i, skip_first_seconds, skip_mitigation_seconds, "constant", const_threshold))
                elif mitigation == "smart":
                    skip_mitigation_seconds = mitigation_configs["skip_mitigation_seconds"]
                    skip_first_seconds = mitigation_configs["skip_first_seconds"]
                    p = Process(target=self.inference_single_episode_rl_mitigation, args=(i, skip_first_seconds, skip_mitigation_seconds))
                else:
                    raise not NotImplementedError
                p.start()
                p.join()
                self.stop_sim()
        except Exception as e:
            print(e)
            self.stop_sim()

    def inference_single_episode(self, iteration):
        self.start_sim()
        self.policy_net.eval()
        if PLOT:
            plt.figure(figsize=(5, 2.5))
        self.carla_rl_env = CarlaRLEnv(config_dict=self.config_dict)
        self.carla_rl_env.prepare_step()
        done = False
        reward_list = list()
        risk_list = list()
        time_list = list()
        prev_points = None
        prev_target_cam = None
        prev_tick_data = None
        first_act = True
        episode_reward = 0
        while not done:
            input_data, timestamp, rewards = self.carla_rl_env.step_sense()  # prev_state + action rewards
            tick_data = self.carla_rl_env.scene_env.manager._agent._agent.tick(input_data)
            img = torchvision.transforms.functional.to_tensor(tick_data['image'])  # current state
            img = img[None].cuda()
            target = torch.from_numpy(tick_data['target'])
            target = target[None].cuda()
            # needs to act
            if first_act:
                self.policy_net.eval()
                points, (target_cam, _) = self.policy_net(img, target)
                first_act = False
                prev_points = torch.FloatTensor(points.cpu())
                prev_target_cam = torch.FloatTensor(target_cam.float().cpu())
                prev_tick_data = copy.deepcopy(tick_data)
                print("Update first action.")
            if rewards[0] is not None and rewards[1] is not None:  # time to update
                self.policy_net.eval()
                points, (target_cam, _) = self.policy_net(img, target)  # act on current state
                prev_points = torch.FloatTensor(points.cpu())  # store current action
                prev_target_cam = torch.FloatTensor(target_cam.float().cpu())  # store current action
                prev_tick_data = copy.deepcopy(tick_data)  # store current action
                print("Update action.")
            action = [prev_points, (prev_target_cam, _), prev_tick_data]
            done = self.carla_rl_env.step_act(timestamp, action)  # use current action
            if rewards[0] is not None and rewards[1] is not None:
                episode_reward += rewards[1]
                print("#############################")
                print("Current risk:", rewards[0])
                print("Current reward:", rewards[1])
                print("#############################")
                risk_list.append(rewards[0])
                time_list.append(int(timestamp.elapsed_seconds * 1e9))
                reward_list.append(rewards[1])
                if PLOT:
                    plt.cla()
                    plt.plot(time_list[:-1], risk_list[:-1], color='r', linewidth=2)
                    plt.plot(time_list[:-1], reward_list[:-1], color='b', linewidth=2)
                    plt.show(block=False)
                    plt.pause(0.025)
            print("Total reward:", episode_reward)
        self.carla_rl_env.post_step()
        del self.carla_rl_env
        print("================= Finish episode {} =================".format(iteration))
        if PLOT:
            plt.close()
        self.stop_sim()

    def inference_single_episode_rl_mitigation(self, episode, skip_first_seconds,
                                               skip_mitigation_seconds, rl_mitigation=True, const_threshold=0.3):
        # set policy net mode
        self.policy_net.eval()

        # set qvalues net mode
        self.load_pytorch_states(0, "inference_dicts")
        self.qvalue_net.eval()

        # setup inference
        self.carla_rl_env = CarlaRLEnv(config_dict=self.config_dict)
        self.carla_rl_env.prepare_step()
        prev_state_list = list()
        done = False
        first_act = True
        reward_list = list()
        risk_list = list()
        time_list = list()
        mitigation_list = list()
        frame_counter = 0
        if PLOT:
            plt.figure(figsize=(3, 2))
        while not done:
            # get data and then apply action
            input_data, timestamp, rewards = self.carla_rl_env.step_sense()
            rewards = list(rewards)
            print("{} seconds in game passed.".format(timestamp.elapsed_seconds))
            tick_data = self.carla_rl_env.scene_env.manager._agent._agent.tick(input_data)
            img = torchvision.transforms.functional.to_tensor(tick_data['image'])
            img = img[None].cuda()
            target = torch.from_numpy(tick_data['target'])
            target = target[None].cuda()
            if first_act:
                # needs to act the first time but without mitigation
                first_act = False
                points, (target_cam, _) = self.policy_net(img, target)
                # print("Update first action.")
            if rewards[0] is not None and rewards[1] is not None:
                self.total_frame_counter += 1
                if timestamp.elapsed_seconds > skip_first_seconds:
                    print("At {} seconds, saving state-action tuples.".format(timestamp.elapsed_seconds))
                    assert len(prev_state_list) >= 3
                    gray_img = torch.stack([self.grayscale_trans(img[0][0:3]),
                                            self.grayscale_trans(img[0][3:6]),
                                            self.grayscale_trans(img[0][6:9])], dim=1)
                    current_prev_three_states = [prev_state_list[-3], prev_state_list[-2], prev_state_list[-1],
                                                 [gray_img, target]]
                    state_imgs = [item[0].squeeze() for item in current_prev_three_states]
                    state_imgs = torch.vstack(state_imgs)
                    state_targets = [item[1].squeeze() for item in current_prev_three_states]
                    state_targets = torch.vstack(state_targets)
                    states = [state_imgs, state_targets]

                    # calculate episode_reward
                    if not np.isnan(rewards[1]):
                        self.episode_reward += rewards[1]
                    else:
                        self.episode_reward += 0
                else:
                    print("Skip conversion, the game just started.")

                # act on current state
                points, (target_cam, _) = self.policy_net(img, target)
                mitigation_action = 0
                # if timestamp.elapsed_seconds > skip_mitigation_seconds and len(
                #         self.replay_buffer) > start_training_replaybuffer_size:
                if timestamp.elapsed_seconds > skip_mitigation_seconds:
                    epsilon = self.epsilon_by_frame(self.total_update_steps)
                    if rl_mitigation and rl_mitigation != "constant":
                        start = time.time()
                        mitigation_action = self.get_action(states, epsilon, frame_counter, risk=rewards[0], inference=True)
                        print("SMC time now:", time.time() - start)
                        # mitigation_action = 1
                        print("Mitigation action:", mitigation_action)
                    elif rl_mitigation and rl_mitigation == "constant":
                        if rewards[0] > const_threshold:
                            mitigation_action = 1
                        else:
                            mitigation_action = 0
                        print("Mitigation action constant threshold:", mitigation_action)
                    else:
                        mitigation_action = 0
                        print("Mitigation action turned off.")
                    points = self.get_mitigation_action(points, mitigation_action)

                ################### Debug ####################
                # self.policy_net.eval()
                # ones = np.ones_like(img.cpu())
                # ones_target = np.ones_like(target.cpu())
                # ones = torch.from_numpy(ones).cuda()
                # ones_target = torch.from_numpy(ones_target).cuda()
                # print(self.policy_net.forward(ones, ones_target, debug=True)[0])
                ################### Debug ####################
                prev_state_list = list()

            else:
                # collect previous state information
                gray_img = torch.stack([self.grayscale_trans(img[0][0:3]),
                                        self.grayscale_trans(img[0][3:6]),
                                        self.grayscale_trans(img[0][6:9])], dim=1)
                states = [gray_img, target]
                prev_state_list.append(copy.deepcopy(states))
            print("Action:", points)
            action = [points, (target_cam, _), tick_data]
            done = self.carla_rl_env.step_act(timestamp, action)

            if rewards[0] is not None and rewards[1] is not None:
                print("#############################")
                print("Current risk:", rewards[0])
                print("Current reward:", rewards[1])
                print("Current mitigation activation:", mitigation_action)
                print("#############################")
                risk_list.append(rewards[0])
                time_list.append(int(timestamp.elapsed_seconds * 1e9))
                if not np.isnan(rewards[1]):
                    reward_list.append(rewards[1])
                else:
                    reward_list.append(0)
                mitigation_list.append(mitigation_action)
                if PLOT:
                    plt.cla()
                    plt.plot(time_list[:-1], risk_list[:-1], color='r', linewidth=2, label="Importance")
                    plt.plot(time_list[:-1], mitigation_list[:-1], color='g', linewidth=2, label="Mitigation activation")
                    plt.xlabel("t (ns)")
                    plt.ylabel("Scene importance")
                    # plt.plot(time_list[:-1], reward_list[:-1], color='b', linewidth=2)
                    plt.ylim(0,1.1)
                    plt.tight_layout()

                    try:
                        pass
                        # plt.show(block=False)   # This plugin does not support raise()
                    except Exception as e:
                        print(e)

                    try:
                        pass
                        # plt.pause(0.001)  # This plugin does not support raise()
                    except Exception as e:
                        print(e)
                writer.add_scalar("Inference Running reward", reward_list[-1], frame_counter)
                writer.add_scalar("Inference Running risk", risk_list[-1], frame_counter)
                writer.add_scalar("Inference Running mitigation activation", mitigation_list[-1], frame_counter)
                writer.flush()
                frame_counter += 1
            print("Total reward:", self.episode_reward)

        if PLOT:
            plt.close()
        self.carla_rl_env.post_step()
        del self.carla_rl_env
        print("================= Finish episode {} =================".format(episode))
        self.all_rewards.append(self.episode_reward)
        self.avg_rewards.append(np.mean(self.all_rewards[-10:]))
        print(
            "Episode: {}, reward: {}, average_reward: {} \n".format(episode, np.round(self.episode_reward, decimals=2),
                                                                    np.mean(self.all_rewards[-10:]))
        )
        if "mitigation_risk_save" in self.config_dict:
            if os.path.isdir(self.config_dict["mitigation_risk_save"]):
                print("Saving mitigation results to {}".format(self.config_dict["mitigation_risk_save"]))
                f = open(os.path.join(self.config_dict["mitigation_risk_save"], "mitigation_dump.pkl"), "wb")
                pickle.dump({
                    "time_list": time_list,
                    "risk_list": risk_list,
                    "mitigation_list": mitigation_list,
                    "reward_list": reward_list
                }, f)
                f.close()
            else:
                print("Save mitigation abort as dir {} does not exist.".format(self.config_dict["mitigation_risk_save"]))

    def inference_step_loop(self):
        self.policy_net.eval()
        for i in range(10):
            self.start_sim()
            if PLOT:
                plt.figure(figsize=(5, 2.5))
            self.carla_rl_env = CarlaRLEnv(config_dict=self.config_dict)
            self.carla_rl_env.prepare_step()
            done = False
            reward_list = list()
            risk_list = list()
            time_list = list()
            prev_points = None
            prev_target_cam = None
            prev_tick_data = None
            first_act = True
            episode_reward = 0
            while not done:
                input_data, timestamp, rewards = self.carla_rl_env.step_sense()  # prev_state + action rewards
                tick_data = self.carla_rl_env.scene_env.manager._agent._agent.tick(input_data)
                img = torchvision.transforms.functional.to_tensor(tick_data['image'])  # current state
                img = img[None].cuda()
                target = torch.from_numpy(tick_data['target'])
                target = target[None].cuda()
                # needs to act
                if first_act:
                    self.policy_net.eval()
                    points, (target_cam, _) = self.policy_net(img, target)
                    first_act = False
                    prev_points = torch.FloatTensor(points.cpu())
                    prev_target_cam = torch.FloatTensor(target_cam.float().cpu())
                    prev_tick_data = copy.deepcopy(tick_data)
                    print("Update first action.")
                if rewards[0] is not None and rewards[1] is not None:  # time to update
                    self.policy_net.eval()
                    points, (target_cam, _) = self.policy_net(img, target)  # act on current state
                    prev_points = torch.FloatTensor(points.cpu())  # store current action
                    prev_target_cam = torch.FloatTensor(target_cam.float().cpu())  # store current action
                    prev_tick_data = copy.deepcopy(tick_data)  # store current action
                    print("Update action.")
                action = [prev_points, (prev_target_cam, _), prev_tick_data]
                done = self.carla_rl_env.step_act(timestamp, action)  # use current action
                if rewards[0] is not None and rewards[1] is not None:
                    episode_reward += rewards[1]
                    print("#############################")
                    print("Current risk:", rewards[0])
                    print("Current reward:", rewards[1])
                    print("#############################")
                    risk_list.append(rewards[0])
                    time_list.append(int(timestamp.elapsed_seconds * 1e9))
                    reward_list.append(rewards[1])
                    if PLOT:
                        plt.cla()
                        plt.plot(time_list[:-1], risk_list[:-1], color='r', linewidth=2)
                        plt.plot(time_list[:-1], reward_list[:-1], color='b', linewidth=2)
                        plt.show(block=False)
                        plt.pause(0.025)
                print("Total reward:", episode_reward)
            self.carla_rl_env.post_step()
            del self.carla_rl_env
            print("================= Finish episode {} =================".format(i))
            if PLOT:
                plt.close()
            self.stop_sim()

    @staticmethod
    def mitigation_emergency_brake(points):
        points[:, :, 1] = 1.0  # hard braking
        return points

    def inference_step_loop_threshold_mitigation(self, risk_threshold=0.1):
        self.policy_net.eval()
        for i in range(1):
            self.start_sim()
            if PLOT:
                plt.figure(figsize=(5, 2.5))
            self.carla_rl_env = CarlaRLEnv(config_dict=self.config_dict)
            self.carla_rl_env.prepare_step()
            done = False
            reward_list = list()
            risk_list = list()
            time_list = list()
            prev_points = None
            prev_target_cam = None
            prev_tick_data = None
            first_act = True
            apply_mitigation_list = list()
            apply_mitigation = 0
            episode_reward = 0
            while not done:
                input_data, timestamp, rewards = self.carla_rl_env.step_sense()
                tick_data = self.carla_rl_env.scene_env.manager._agent._agent.tick(input_data)
                img = torchvision.transforms.functional.to_tensor(tick_data['image'])
                img = img[None].cuda()
                target = torch.from_numpy(tick_data['target'])
                target = target[None].cuda()
                # needs to act
                if first_act:
                    self.policy_net.eval()
                    points, (target_cam, _) = self.policy_net(img, target)
                    first_act = False
                    prev_points = torch.FloatTensor(points.cpu())
                    prev_target_cam = torch.FloatTensor(target_cam.float().cpu())
                    prev_tick_data = copy.deepcopy(tick_data)
                    print("Update first action.")
                if rewards[0] is not None and rewards[1] is not None:
                    self.policy_net.eval()
                    points, (target_cam, _) = self.policy_net(img, target)
                    prev_points = torch.FloatTensor(points.cpu())
                    prev_target_cam = torch.FloatTensor(target_cam.float().cpu())
                    prev_tick_data = copy.deepcopy(tick_data)
                    print("Update action.")
                if rewards[0] is not None and rewards[0] < risk_threshold:
                    apply_mitigation = 0
                if (rewards[0] is not None and rewards[0] >= risk_threshold) or apply_mitigation > 0.05:
                    self.mitigation_emergency_brake(prev_points)
                    apply_mitigation = 0.1
                action = [prev_points, (prev_target_cam, _), prev_tick_data]
                done = self.carla_rl_env.step_act(timestamp, action)
                if rewards[0] is not None and rewards[1] is not None:
                    episode_reward += rewards[1]
                    print("#############################")
                    print("Current risk:", rewards[0])
                    print("Current reward:", rewards[1])
                    print("Current mitigation activation:", apply_mitigation)
                    print("#############################")
                    risk_list.append(rewards[0])
                    time_list.append(int(timestamp.elapsed_seconds * 1e9))
                    reward_list.append(rewards[1])
                    apply_mitigation_list.append(apply_mitigation)
                    if PLOT:
                        plt.cla()
                        plt.plot(time_list[:-1], risk_list[:-1], color='r', linewidth=2)
                        plt.plot(time_list[:-1], apply_mitigation_list[:-1], color='g', linewidth=2)
                        plt.plot(time_list[:-1], reward_list[:-1], color='b', linewidth=2)
                        plt.show(block=False)
                        plt.pause(0.001)
                print("Total reward:", episode_reward)
            self.carla_rl_env.post_step()
            del self.carla_rl_env
            print("================= Finish episode {} =================".format(i))
            if PLOT:
                plt.close()
            self.stop_sim()

    def update_agent(self, batch_size):
        """
        define new tuple pair:
        1. states: B x (1 current + 3 previous frames) * 3 cameras x H x W
        2. actions: B x 1 x 4
        3. rewards: B x 1 x 1
        4. next_states: B x (1 next current + 3 next previous frames) * 3 cameras x H x W
        5. done: B x 1 x 1 
        """
        self.policy_net.eval()
        self.qvalue_net.eval()
        self.qvalue_net_target.eval()
        states, actions, rewards, next_states, done = self.replay_buffer.random_draw(batch_size)

        imgs = [item[0] for item in states]
        imgs = torch.stack(imgs)
        imgs = torch.squeeze(imgs)
        targets = [item[1] for item in states]
        targets = torch.stack(targets)
        targets = torch.squeeze(targets)
        rewards = [item[1] for item in rewards]
        rewards = (torch.Tensor(rewards)).unsqueeze(dim=1)
        done = (torch.LongTensor(done)).unsqueeze(dim=1)
        actions = (torch.LongTensor(actions)).unsqueeze(dim=1)
        imgs = imgs.cuda()
        targets = targets.cuda()
        actions = actions.cuda()
        rewards = rewards.cuda()
        done = done.cuda()

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=torch.device("cuda"),
                                      dtype=torch.uint8)
        try:
            non_final_next_states = [[s[0], s[1]] for s in next_states if s is not None]
            assert len(non_final_next_states) > 0
            next_imgs = [item[0] for item in non_final_next_states]
            next_imgs = torch.stack(next_imgs)
            next_imgs = torch.squeeze(next_imgs)
            next_targets = [item[1] for item in non_final_next_states]
            next_targets = torch.stack(next_targets)
            next_targets = torch.squeeze(next_targets)
            next_imgs = next_imgs.cuda()
            next_targets = next_targets.cuda()
            empty_next_state_values = False
        except:
            print("All terminal states.")
            next_imgs = None
            next_targets = None
            empty_next_state_values = True

        # calculate the Qvalues
        current_qvalues = self.qvalue_net(imgs, targets).gather(1, actions)
        with torch.no_grad():
            max_next_qvalues = torch.zeros(batch_size, device=torch.device("cuda"), dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action_double_dqn(next_imgs, next_targets).long()
                max_next_qvalues[non_final_mask] = self.qvalue_net_target(next_imgs, next_targets).gather(1,
                                                                                                          max_next_action)
            expected_qvalues = rewards + (self.config_dict["reward_decay"] * max_next_qvalues)

        # update qvalues model
        qvalue_net_loss = self.qvalue_net_criterion(current_qvalues, expected_qvalues)
        writer.add_scalar("Qvalue net loss", qvalue_net_loss.cpu().detach().item(), self.total_update_steps)
        writer.flush()
        self.qvalue_net_optimizer.zero_grad()
        qvalue_net_loss.backward()
        # clip the graident for better stability
        for param in self.qvalue_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.qvalue_net_optimizer.step()

        # hard/soft update target network weights
        self.hard_update_target_model()
        # self.soft_update_target_model()

        # update book keeping information
        self.total_update_steps += 1

    def hard_update_target_model(self):
        print("Hard update target network")
        if self.total_update_steps > 0 and self.total_update_steps % self.target_net_update_freq == 0:
            self.qvalue_net_target.load_state_dict(self.qvalue_net.state_dict())

    def soft_update_target_model(self):
        print("Soft update target network with tau:", self.config_dict["tau"])
        for target_param, param in zip(self.qvalue_net_target.parameters(), self.qvalue_net.parameters()):
            target_param.data.copy_(
                param.data * self.config_dict["tau"] + target_param.data * (1.0 - self.config_dict["tau"]))

    def get_max_next_state_action_double_dqn(self, next_imgs, next_targets):
        return self.qvalue_net(next_imgs, next_targets).max(dim=1)[1].view(-1, 1)

    def get_action(self, states, epsilon, frame_counter, risk, inference=False):
        with torch.no_grad():
            if self.second_action:
                self.second_action = False
                print("Repeating previous action.")
                return self.previous_action
            if random.random() >= epsilon * 2 or inference:
                imgs = states[0]
                targets = states[1]
                imgs = imgs[None, :]
                targets = targets[None, :]
                self.qvalue_net.eval()
                actions = self.qvalue_net(imgs, targets).max(1)[1].view(1, 1)
                writer.add_scalar("Random selection", 0, frame_counter)
                writer.flush()
                self.second_action = True
                self.previous_action = actions.item()

                if risk > 0.05:
                    return actions.item()
                else:
                    self.second_action = True
                    self.previous_action = 0
                    return 0
            else:
                if self.config_dict["action_space"] == 2:
                    actions_list = [0] * 7 + [1] * 3
                    # actions_list = [0] * 3 + [1] * 7
                actions = random.choice(actions_list)
                if risk <= 1e-3:
                    actions = 0
                writer.add_scalar("Random selection", 1, frame_counter)
                writer.flush()
                writer.add_scalar("Random action", actions, frame_counter)
                writer.flush()
                self.second_action = True
                self.previous_action = actions
                return actions

    def get_mitigation_action(self, points, action_mode):
        action_mode = self.ACTION_MAP[action_mode]
        modified_action = points.cpu().clone().detach()
        if action_mode == "NOOP":
            pass
        elif action_mode == "EB":
            modified_action[:, :, 1] = 1.0  # hard braking
        elif action_mode == "MR":
            raise NotImplementedError
        elif action_mode == "ML":
            raise NotImplementedError
        else:
            raise RuntimeError("Unsupported action.")
        return modified_action

    @staticmethod
    def epsilon_by_frame(frames, epsilon_final=0.01, epsilon_start=0.9, epsilon_decay=3000):
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1.0 * frames / epsilon_decay)
        writer.add_scalar("Epsilon exploration", epsilon, frames)
        writer.flush()
        return epsilon

    def start_sim(self):
        self.stop_sim()
        subprocess.Popen(self.sim_start)
        time.sleep(3)

    def stop_sim(self):
        subprocess.Popen(self.sim_kill)
        time.sleep(2)

    def load_replay_buffer(self, episode):
        if episode >= 0:
            print("New episode load replay buffer.")
            replay_buffer_pickle = os.path.join(self.ram_shared_path, "replay_buffer.pkl".format(episode))
            f = open(replay_buffer_pickle, "rb")
            self.replay_buffer = pickle.load(f)
            f.close()
        else:
            print("Nothing to load for the first episode.")

    def save_replay_buffer(self, episode):
        if episode > 0:
            print("Finish episode save replay buffer.")
            old_replay_buffer_pickle = os.path.join(self.ram_shared_path, "replay_buffer_old.pkl")
            if os.path.isfile(old_replay_buffer_pickle):
                os.remove(old_replay_buffer_pickle)
            os.rename(os.path.join(self.ram_shared_path, "replay_buffer.pkl"), os.path.join(self.ram_shared_path, "replay_buffer_old.pkl"))

        replay_buffer_pickle = os.path.join(self.ram_shared_path, "replay_buffer.pkl")


        f = open(replay_buffer_pickle, "wb")
        pickle.dump(self.replay_buffer, f)
        f.close()

    def load_pytorch_states(self, episode, prefix="state_dicts"):
        if episode >= 0:
            print("New episode load state dicts.")
            state_dicts_pickle = os.path.join(self.ram_shared_path, "{}.{}.pkl".format(prefix, episode))
            f = open(state_dicts_pickle, "rb")
            state_dicts = pickle.load(f)
            f.close()
            self.qvalue_net.load_state_dict(state_dicts["qvalue_net"])
            self.qvalue_net_target.load_state_dict(state_dicts["qvalue_net_target"])
            self.total_update_steps = state_dicts["total_update_steps"]
            self.qvalue_net_optimizer.load_state_dict(state_dicts["qvalue_net_optimizer"])
            return state_dicts
        else:
            print("Nothing to load for the first episode.")
            return None

    def save_pytorch_states(self, episode):
        print("Finish episode save state dicts.")
        state_dicts_pickle = os.path.join(self.ram_shared_path, "state_dicts.{}.pkl".format(episode))
        f = open(state_dicts_pickle, "wb")
        pickle.dump({
            "qvalue_net": self.qvalue_net.state_dict(),
            "qvalue_net_target": self.qvalue_net_target.state_dict(),
            "qvalue_net_optimizer": self.qvalue_net_optimizer.state_dict(),
            "total_update_steps": self.total_update_steps,
            "replay_buffer_cleared": self.replay_buffer_cleared
        }, f)
        f.close()

    def load_bookkeeping_info(self, episode):
        print("Load episode book keeping information.")
        if episode >= 0:
            book_keeping_pickle = os.path.join(self.ram_shared_path, "book_keeping.{}.pkl".format(episode))
            f = open(book_keeping_pickle, "rb")
            save_dict = pickle.load(f)
            f.close()
        else:
            print("No bookkeeping information to load at the first episode.")
            return None
        return save_dict

    def save_bookkeeping_info(self, episode, save_dict):
        print("Finish episode save book keeping information.")
        book_keeping_pickle = os.path.join(self.ram_shared_path, "book_keeping.{}.pkl".format(episode))
        f = open(book_keeping_pickle, "wb")
        pickle.dump(save_dict, f)
        f.close()

    def training_loop_standalone_process(
            self, total_episodes, batch_size, start_training_replaybuffer_size, skip_first_seconds, mitigation_penalty,
            skip_mitigation_seconds, save_frequency, resume=0, target_net_update_freq=100, clear_replay_buffer=False, load_checkpoint=-1):
        self.all_rewards = []
        self.avg_rewards = []
        self.total_frame_counter = 0
        self.target_net_update_freq = target_net_update_freq
        if load_checkpoint != -1:
            self.load_checkpoint(load_checkpoint)
        for episode in range(resume, total_episodes):
            self.episode_reward = 0
            self.start_sim()
            p = Process(
                target=self.training_single_episode_reinforcement_mitigation, args=(
                    episode, batch_size, start_training_replaybuffer_size, skip_first_seconds, skip_mitigation_seconds,
                    save_frequency, mitigation_penalty, clear_replay_buffer)
            )
            p.start()
            p.join()
            self.stop_sim()
            self.first_run = False
            save_dict = self.load_bookkeeping_info(episode)
            writer.add_scalar("Episode rewards", save_dict["episode_reward"], episode)
            writer.flush()
            self.all_rewards.append(save_dict["episode_reward"])
            self.avg_rewards.append(np.mean(self.all_rewards[-10:]))

        print("Done training {} episodes.".format(total_episodes - 1))

        self.save_checkpoint("final", self.all_rewards)
        writer.close()

    def training_single_episode_reinforcement_mitigation(
            self, episode, batch_size, start_training_replaybuffer_size, skip_first_seconds,
            skip_mitigation_seconds, save_frequency, mitigation_penalty, clear_replay_buffer=False):
        assert start_training_replaybuffer_size >= batch_size

        # load previous episode states:
        self.load_replay_buffer(episode - 1)
        state_dicts = self.load_pytorch_states(episode - 1)
        try:
            self.replay_buffer_cleared = state_dicts["replay_buffer_cleared"]
        except Exception:
            print("Failed to read replay_buffer_cleared variable, default to False.")
            self.replay_buffer_cleared = False
        if clear_replay_buffer and (not self.replay_buffer_cleared or self.first_run):
            print("Size before clear:", len(self.replay_buffer))
            # for i in range(int(len(self.replay_buffer) * 0.5)):
            nums_to_pop = int(len(self.replay_buffer) - self.config_dict["replay_buffer_size"])
            for i in range(max(nums_to_pop, 0)):
                if random.random() > 0.5:
                    self.replay_buffer.memory.popleft()
                else:
                    self.replay_buffer.memory.pop()
            print("Size after clear:", len(self.replay_buffer))
            self.replay_buffer_cleared = True
        save_dict = self.load_bookkeeping_info(episode - 1)
        if save_dict is not None:
            self.total_frame_counter = save_dict["total_frames"]
        else:
            self.total_frame_counter = 0

        self.carla_rl_env = CarlaRLEnv(config_dict=self.config_dict)
        self.carla_rl_env.prepare_step()
        prev_state_list = list()
        prev_states = None
        done = False
        prev_action = None
        first_act = True
        reward_list = list()
        risk_list = list()
        time_list = list()
        mitigation_list = list()
        frame_counter = 0
        if PLOT:
            plt.figure(figsize=(5, 2.5))
        while not done:
            # get data and then apply action
            input_data, timestamp, rewards = self.carla_rl_env.step_sense()
            rewards = list(rewards)
            print("{} seconds in game passed.".format(timestamp.elapsed_seconds))
            tick_data = self.carla_rl_env.scene_env.manager._agent._agent.tick(input_data)
            img = torchvision.transforms.functional.to_tensor(tick_data['image'])
            img = img[None].cuda()
            target = torch.from_numpy(tick_data['target'])
            target = target[None].cuda()
            if first_act:
                # needs to act the first time but without mitigation
                first_act = False
                points, (target_cam, _) = self.policy_net(img, target)
                prev_action = 0
                prev_states = None
                # print("Update first action.")
            if rewards[0] is not None and rewards[1] is not None:
                pushed = False
                updated = False
                self.total_frame_counter += 1
                if timestamp.elapsed_seconds > skip_first_seconds:
                    print("At {} seconds, saving state-action tuples.".format(timestamp.elapsed_seconds))
                    assert len(prev_state_list) >= 3
                    gray_img = torch.stack([self.grayscale_trans(img[0][0:3]),
                                            self.grayscale_trans(img[0][3:6]),
                                            self.grayscale_trans(img[0][6:9])], dim=1)
                    current_prev_three_states = [prev_state_list[-3], prev_state_list[-2], prev_state_list[-1],
                                                 [gray_img, target]]
                    state_imgs = [item[0].squeeze() for item in current_prev_three_states]
                    state_imgs = torch.vstack(state_imgs)
                    state_targets = [item[1].squeeze() for item in current_prev_three_states]
                    state_targets = torch.vstack(state_targets)
                    states = [state_imgs, state_targets]
                    try:
                        memory_blob = [
                            copy.deepcopy([prev_states[0].cpu(), prev_states[1].cpu().float()]),
                            prev_action,
                            copy.deepcopy(rewards),
                            copy.deepcopy([states[0].cpu(), states[1].cpu().float()]), copy.deepcopy(int(done))
                        ]
                        if not np.isnan(rewards[1]) and prev_states is not None and not len(prev_states[0]) == 1:
                            self.replay_buffer.add(memory_blob)
                            pushed = True
                    except Exception:
                        print("Push failed, memory blob:", prev_states, prev_action, rewards, states)

                    # calculate episode_reward
                    if not np.isnan(rewards[1]):
                        self.episode_reward += rewards[1]
                        updated = True
                    else:
                        self.episode_reward += 0
                        updated = True
                else:
                    print("Skip save, the game just started.")

                # act on current state

                points, (target_cam, _) = self.policy_net(img, target)

                mitigation_action = 0
                # if timestamp.elapsed_seconds > skip_mitigation_seconds and len(
                #         self.replay_buffer) > start_training_replaybuffer_size:
                if timestamp.elapsed_seconds > skip_mitigation_seconds:
                    epsilon = self.epsilon_by_frame(self.total_update_steps)
                    start = time.time()
                    mitigation_action = self.get_action(states, epsilon, frame_counter, risk=rewards[0])
                    print("Mitigation action:", mitigation_action)
                    points = self.get_mitigation_action(points, mitigation_action)

                    print("SMC time now:", time.time() - start)

                    # change the step reward according to mitigation
                    if mitigation_action != 0 and rewards[0] <= 1e-3 and (pushed or updated):
                        rewards[1] -= mitigation_penalty
                        if pushed:
                            self.replay_buffer.memory[-1][2] = list(self.replay_buffer.memory[-1][2])
                            self.replay_buffer.memory[-1][2][1] -= mitigation_penalty
                        if updated:
                            self.episode_reward -= mitigation_penalty

                ################### Debug ####################
                # self.policy_net.eval()
                # ones = np.ones_like(img.cpu())
                # ones_target = np.ones_like(target.cpu())
                # ones = torch.from_numpy(ones).cuda()
                # ones_target = torch.from_numpy(ones_target).cuda()
                # print(self.policy_net.forward(ones, ones_target, debug=True)[0])
                ################### Debug ####################

                prev_action = mitigation_action  # store current action
                prev_states = states  # store current state as previous state
                prev_state_list = list()

                # step the trainer
                if len(self.replay_buffer) > start_training_replaybuffer_size:
                    print("!!!!!!!!!!!!!!!!!! Update state: {} !!!!!!!!!!!!!!!!!!".format(self.total_update_steps))
                    self.update_agent(batch_size)
                else:
                    print("+++++++++++ Replay buffer size: {} +++++++++++".format(len(self.replay_buffer)))
            else:
                # collect previous state information
                gray_img = torch.stack([self.grayscale_trans(img[0][0:3]),
                                        self.grayscale_trans(img[0][3:6]),
                                        self.grayscale_trans(img[0][6:9])], dim=1)
                states = [gray_img, target]
                prev_state_list.append(copy.deepcopy(states))
            print("Action:", points)
            action = [points, (target_cam, _), tick_data]
            done = self.carla_rl_env.step_act(timestamp, action)

            if rewards[0] is not None and rewards[1] is not None:
                print("#############################")
                print("Current risk:", rewards[0])
                print("Current reward:", rewards[1])
                print("Current mitigation activation:", mitigation_action)
                print("#############################")
                risk_list.append(rewards[0])
                time_list.append(int(timestamp.elapsed_seconds * 1e9))
                if not np.isnan(rewards[1]):
                    reward_list.append(rewards[1])
                else:
                    reward_list.append(0)
                mitigation_list.append(mitigation_action)
                if PLOT:
                    plt.cla()
                    plt.plot(time_list[:-1], risk_list[:-1], color='r', linewidth=2)
                    plt.plot(time_list[:-1], mitigation_list[:-1], color='g', linewidth=2)
                    plt.plot(time_list[:-1], reward_list[:-1], color='b', linewidth=2)
                    plt.show(block=False)
                    plt.pause(0.001)
                writer.add_scalar("Running reward", reward_list[-1], frame_counter)
                writer.add_scalar("Running risk", risk_list[-1], frame_counter)
                writer.add_scalar("Running mitigation activation", mitigation_list[-1], frame_counter)
                writer.flush()
                frame_counter += 1
            print("Total reward:", self.episode_reward)

            if done:
                # write the final memory
                try:
                    if rewards[1] is not None:
                        if not np.isnan(rewards[1]) and prev_states is not None and not len(prev_states[0]) == 1:
                            memory_blob = [
                                copy.deepcopy([prev_states[0].cpu(), prev_states[1].cpu().float()]),
                                prev_action,
                                copy.deepcopy(rewards), None, copy.deepcopy(int(done))
                            ]
                            print("Final step reward is:", rewards)
                            self.replay_buffer.add(memory_blob)
                    else:
                        if prev_states is not None and not len(prev_states[0]) == 1:
                            rewards = list(rewards)
                            rewards[1] = reward_list[-1]
                            rewards = tuple(rewards)
                            memory_blob = [
                                copy.deepcopy([prev_states[0].cpu(), prev_states[1].cpu().float()]),
                                prev_action,
                                copy.deepcopy(rewards), None, copy.deepcopy(int(done))
                            ]
                            print("Final step approx. reward is:", rewards)
                            self.replay_buffer.add(memory_blob)
                except Exception:
                    print("Push failed, memory blob:", prev_states, prev_action, rewards, states)
        if PLOT:
            plt.close()
        self.carla_rl_env.post_step()
        del self.carla_rl_env
        print("================= Finish episode {} =================".format(episode))
        self.all_rewards.append(self.episode_reward)
        self.avg_rewards.append(np.mean(self.all_rewards[-10:]))
        print(
            "Episode: {}, reward: {}, average_reward: {} \n".format(episode, np.round(self.episode_reward, decimals=2),
                                                                    np.mean(self.all_rewards[-10:]))
        )
        # save weight
        if episode % save_frequency == 0:
            self.save_checkpoint(episode, self.all_rewards)

        # save current episode states:
        self.save_replay_buffer(episode)
        self.save_pytorch_states(episode)
        self.save_bookkeeping_info(episode, {
            "episode_reward": self.episode_reward,
            "reward_list": reward_list,
            "risk_list": risk_list,
            "time_list": time_list,
            "mitigation_list": mitigation_list,
            "episode_frames": frame_counter,
            "total_frames": self.total_frame_counter
        })

    def save_checkpoint(self, episode, rewards):
        save_name = "episode_{}_mitigation_rl.pth".format(episode)
        save_path = os.path.join(self.config_dict["rl_checkpoint_savedir"], save_name)
        torch.save({
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'qvalue_net_state_dict': self.qvalue_net.state_dict(),
            'qvalue_net_target_state_dict': self.qvalue_net_target.state_dict(),
            'qvalue_net_optimizer_state_dict': self.qvalue_net_optimizer.state_dict(),
            'rewards': rewards
        }, save_path)

    def load_checkpoint(self, episode):
        self.load_pytorch_states(episode)
        self.total_update_steps = 0
        self.total_frame_counter = 0


def main():
    torch.multiprocessing.set_start_method("spawn")
    dqn_lbc = CarlaLBCDQN()

    dqn_lbc.training_loop_standalone_process(total_episodes=100,
                                             batch_size=32,
                                             start_training_replaybuffer_size=256,
                                             skip_first_seconds=3,
                                             skip_mitigation_seconds=4,
                                             save_frequency=10,
                                             resume=0,
                                             target_net_update_freq=10000,
                                             mitigation_penalty=0.0,
                                             clear_replay_buffer=True,
                                             load_checkpoint=-1)
    # dqn_lbc.inference_loop_standalone_process(total_episodes=1, mitigation="none", mitigation_configs={
    #     "skip_mitigation_seconds": 5,
    #     "skip_first_seconds": 4
    # })


if __name__ == "__main__":
    main()
