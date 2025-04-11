import os.path
import time
import copy
from torch import nn
import torch.optim as optim
import torch
import torchvision
import matplotlib.pyplot as plt
from carla_project.src.image_model import ImageModel
from carla_project.src.actor_critic_model import CriticModel
from carla_project.src.actor_critic_model import PolicyModel
from leaderboard.env_interactor import CarlaRLEnv
from leaderboard.replay_buffer import ReplayBuffer
from torch.autograd import Variable
import numpy as np
from torchviz import  make_dot

PLOT = True


class CarlaLBCDDPG(object):
    def __init__(self):
        self.config_dict = {
            'host': 'localhost',
            'port': '2000',
            'trafficManagerPort': '8000',
            'trafficManagerSeed': 0,
            'debug': 0,
            'record': '',
            'timeout': 3600,
            'sim_data_save': '/media/sheng/data4/projects/DiverseEnv/auto/sim_data_collection',
            'routes': '/media/sheng/data4/projects/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard/data/routes_fi/route_highway.xml',
            'scenarios': '/media/sheng/data4/projects/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard/data/fi_ghost_cutin.json',
            'repetitions': 1,
            'agent': 'image_agent_rl.py',
            'agent_config': '/media/sheng/data4/projects/DiverseEnv/auto/agents/2020_CARLA_challenge/epoch24.ckpt',
            'track': 'SENSORS',
            'resume': False,
            'checkpoint': './simulation_results.json',
            'dual_agent': False,
            'dual_dup': False,
            'log_path': '/media/sheng/data4/projects/DiverseEnv/auto/sim_data_logging',
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
            'risk_evaluation_mode': "nn",
            'risk_pickle_stored': "/media/sheng/data4/projects/DiverseEnv/auto/sim_risk_analyzed/risk_lookup_3_risk_analysis.pkl",
            'traj_pickle_stored': "/media/sheng/data4/projects/DiverseEnv/carladataset/carla-sim/safety-critical-simdata/route_highway.xml_fi_lead_cutin.json.pkl",
            'inference_model_path': "/media/sheng/data4/projects/ReachML/reachml/checkpoints/sim_reduced_cvctr_3s_2022-04-25-00-34-02.pth",
            'inference_config_path': "/media/sheng/data4/projects/ReachML/reachml/data/train_test_sim_lite/route_highway.xml_fi_ghost_cutin.json.pkl/model_unc/data_realtime_trail0_3s/realtime_trail0_3s_data_config.json",
            'actor_lr': 1e-5,
            'critic_lr': 1e-5,
            'reward_decay': 0.99,
            'tau': 1e-1,
            "rl_checkpoint_savedir": "/media/sheng/data4/projects/DiverseEnv/auto/rl_checkpoint"
        }
        self.carla_rl_env = None
        self.checkpoint_dict = torch.load(self.config_dict["agent_config"])
        self.hparams = self.checkpoint_dict["hparams"]

        # insert LBC policy net
        self.policy_net = PolicyModel(self.hparams)
        self.policy_net.load_state_dict(self.checkpoint_dict["state_dict"], strict=False)
        self.policy_net.cuda()
        self.policy_net_target = PolicyModel(self.hparams)
        for target_param, param in zip(self.policy_net_target.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.policy_net_target.cuda()

        # self.policy_net = ImageModel.load_from_checkpoint(self.config_dict["agent_config"])
        # self.policy_net.cuda()
        # self.policy_net_target = ImageModel.load_from_checkpoint(self.config_dict["agent_config"])
        # for target_param, param in zip(self.policy_net_target.parameters(), self.policy_net.parameters()):
        #     target_param.data.copy_(param.data)
        # self.policy_net_target.cuda()

        # self.hparams = copy.deepcopy(self.policy_net.hparams)

        # insert LBC-based Qvalue net
        self.qvalue_net = CriticModel(self.hparams)
        self.qvalue_net.cuda()
        self.qvalue_net_target = CriticModel(self.hparams)
        for target_param, param in zip(self.qvalue_net_target.parameters(), self.qvalue_net.parameters()):
            target_param.data.copy_(param.data)
        self.qvalue_net_target.cuda()

        # insert replay buffer
        self.replay_buffer = ReplayBuffer(10000)

        # insert critic_criterion and optimizers
        self.critic_criterion = nn.MSELoss()
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config_dict["actor_lr"])
        self.critic_optimizer = optim.Adam(self.qvalue_net.parameters(), lr=self.config_dict["critic_lr"])

    def inference_step_loop(self):
        self.policy_net.eval()
        for i in range(1):
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
            print("==================================== Finish episode {} ====================================".format(i))
            if PLOT:
                plt.close()

    @staticmethod
    def mitigation_emergency_brake(points):
        points[:, :, 1] = 1.0  # hard braking
        return points

    def inference_step_loop_threshold_mitigation(self, risk_threshold=0.1):
        self.policy_net.eval()
        for i in range(1):
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
            time.sleep(1)

    @staticmethod
    def inject_action_noise(points, mode="four-dim",
                            scales=(0.0, 0.0), scale_factors=(1.0, 1.1, 1.2, 1.3)):
        assert mode in ["brake", "four-dim"]
        modified_action = points.cpu().clone().detach()
        if mode == "brake" or mode == "four-dim":
            noise = np.random.normal(0.0, scales[0], 1)
            modified_action[0, 0, 1] += noise * scale_factors[0]
            modified_action[0, 1, 1] += noise * scale_factors[1]
            modified_action[0, 2, 1] += noise * scale_factors[2]
            modified_action[0, 3, 1] += noise * scale_factors[3]
        if mode == "four-dim":
            noise = np.random.normal(0.0, scales[1], 1)
            modified_action[0, 0, 0] += noise * scale_factors[0]
            modified_action[0, 1, 0] += noise * scale_factors[1]
            modified_action[0, 2, 0] += noise * scale_factors[2]
            modified_action[0, 3, 0] += noise * scale_factors[3]
        modified_action = np.clip(modified_action, -1.0, 1.0)
        modified_action = torch.tensor(modified_action).cuda()
        print("Original points:", points)
        print("Updated points:", modified_action)
        return modified_action

    def update_agent(self, batch_size):
        self.policy_net.eval()
        self.qvalue_net.eval()
        states, actions, rewards, next_states, done = self.replay_buffer.random_draw(batch_size)
        imgs = [item[0] for item in states]
        imgs = torch.stack(imgs)
        imgs = torch.squeeze(imgs)
        targets = [item[1] for item in states]
        targets = torch.stack(targets)
        targets = torch.squeeze(targets)
        rewards = [item[1] for item in rewards]
        rewards = (torch.Tensor(rewards)).unsqueeze(dim=1)
        next_imgs = [item[0] for item in next_states]
        next_imgs = torch.stack(next_imgs)
        next_imgs = torch.squeeze(next_imgs)
        next_targets = [item[1] for item in next_states]
        next_targets = torch.stack(next_targets)
        next_targets = torch.squeeze(next_targets)
        ego_actions = [item[0] for item in actions]
        ego_actions = torch.stack(ego_actions)
        ego_actions = torch.squeeze(ego_actions)
        done = (torch.IntTensor(done)).unsqueeze(dim=1)
        imgs = imgs.cuda()
        targets = targets.cuda()
        next_imgs = next_imgs.cuda()
        next_targets = next_targets.cuda()
        actions = ego_actions.cuda()
        rewards = rewards.cuda()
        done = done.cuda()

        # calculate critic loss
        qvalues = self.qvalue_net(imgs, targets, actions)
        next_points, (next_target_cam, next_) = self.policy_net_target(next_imgs, next_targets)
        next_ego_actions = next_points
        next_ego_actions = next_ego_actions.cuda()
        next_qvalues = self.qvalue_net_target(next_imgs, next_targets, next_ego_actions.detach())
        q_targeted = rewards + self.config_dict["reward_decay"] * next_qvalues * (1 - done)
        critic_loss = self.critic_criterion(qvalues, q_targeted)
        # z = make_dot(critic_loss, params=dict(self.qvalue_net.named_parameters()),  show_attrs=True, show_saved=True)
        # z.view()

        # calculate actor loss ascend
        # actor_action = self.policy_net(imgs, targets)[0].mean()
        # k = make_dot(dummy_action, params=dict(self.policy_net.named_parameters()), show_attrs=True, show_saved=True)
        # k.view()

        actor_action = self.policy_net(imgs, targets)[0]
        # policy_loss = -self.qvalue_net(imgs, targets, actor_action).mean()
        # s = make_dot(policy_loss, params=dict(self.policy_net.named_parameters()))
        # s.view()
        # ###################################
        # ones_action = np.ones_like(actions.cpu())
        # ones_action = torch.from_numpy(ones_action).cuda()
        # policy_loss = -self.qvalue_net(imgs, targets, ones_action).mean()
        # ###################################

        # parameters_dict = dict()
        # for name, param in self.policy_net.net.named_parameters():
        #     parameters_dict[name] = param.cpu().clone().detach()

        # update network weights
        # self.policy_optimizer.zero_grad()
        # policy_loss.backward()
        # self.policy_optimizer.step()

        # for name, param in self.policy_net.net.named_parameters():
        #     assert name in parameters_dict
        #     old_param = parameters_dict[name]
        #     if not torch.equal(old_param, param.cpu().clone().detach()):
        #         print(name)
        #         print(param)
        #         print(old_param)
        #         pass

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target network weights
        for target_param, param in zip(self.policy_net_target.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data * self.config_dict["tau"] + target_param.data * (1.0 - self.config_dict["tau"]))

        for target_param, param in zip(self.qvalue_net_target.parameters(), self.qvalue_net.parameters()):
            target_param.data.copy_(param.data * self.config_dict["tau"] + target_param.data * (1.0 - self.config_dict["tau"]))

    def training_step_loop_reinforcement_mitigation(self, episodes, batch_size, train_size, skip_first_seconds, save_frequency):
        assert train_size >= batch_size
        all_rewards = []
        avg_rewards = []
        for episode in range(episodes):
            episode_reward = 0
            self.carla_rl_env = CarlaRLEnv(config_dict=self.config_dict)
            self.carla_rl_env.prepare_step()
            prev_states = None
            done = False
            prev_points = None
            prev_target_cam = None
            prev_tick_data = None
            first_act = True
            reward_list = list()
            risk_list = list()
            time_list = list()
            while not done:
                # get data and then apply action
                input_data, timestamp, rewards = self.carla_rl_env.step_sense()
                print("{} seconds in game passed.".format(timestamp.elapsed_seconds))
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
                    prev_states = copy.deepcopy([img, target])
                    # print("Update first action.")
                if rewards[0] is not None and rewards[1] is not None:
                    if timestamp.elapsed_seconds > skip_first_seconds:
                        print("at {} seconds, saving state-action tuples.".format(timestamp.elapsed_seconds))
                        states = [img, target]
                        memory_blob = [
                            copy.deepcopy([prev_states[0].cpu().float(), prev_states[1].cpu().float()]),
                            copy.deepcopy([torch.tensor(prev_points), (prev_target_cam, _.cpu()), prev_tick_data]),
                            copy.deepcopy(rewards), copy.deepcopy(states), copy.deepcopy(int(done))
                        ]
                        self.replay_buffer.add(memory_blob)

                        # calculate episode_reward
                        episode_reward += rewards[1]
                    else:
                        print("Skip save, the game just started.")

                    # print("Update action.")
                    self.policy_net.eval()
                    points, (target_cam, _) = self.policy_net(img, target)  # act on current state

                    ################## Debug ####################
                    self.policy_net.eval()
                    ones = np.ones_like(img.cpu())
                    ones_target = np.ones_like(target.cpu())
                    ones = torch.from_numpy(ones).cuda()
                    ones_target = torch.from_numpy(ones_target).cuda()
                    print(self.policy_net.forward(ones, ones_target, debug=True)[0])
                    ################## Debug ####################

                    if timestamp.elapsed_seconds > skip_first_seconds:
                        points = self.inject_action_noise(points)  # inject noise to action
                    prev_points = torch.FloatTensor(points.cpu())  # store current action
                    prev_target_cam = torch.FloatTensor(target_cam.float().cpu())  # store current action
                    prev_tick_data = copy.deepcopy(tick_data)  # store current action
                    prev_states = copy.deepcopy([img, target])  # store current state as previous state

                    # step the trainer
                    if len(self.replay_buffer) > train_size:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!! Update !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        self.update_agent(batch_size)
                print("Action:", points)
                action = [prev_points, (prev_target_cam, _), prev_tick_data]
                done = self.carla_rl_env.step_act(timestamp, action)
            self.carla_rl_env.post_step()
            del self.carla_rl_env
            print(
                "=============================== Finish episode {} ===============================".format(episode))
            all_rewards.append(episode_reward)
            avg_rewards.append(np.mean(all_rewards[-10:]))
            print(
                "Episode: {}, reward: {}, average_reward: {} \n".format(episode, np.round(episode_reward, decimals=2),
                                                                        np.mean(all_rewards[-10:]))
                )
            # save weight
            if episode % save_frequency == 0:
                self.save_checkpoint(episode, all_rewards)
        print("Doen training {} episodes.".format(episodes))
        self.save_checkpoint("final", all_rewards)

    def save_checkpoint(self, episode, rewards):
        save_name = "episode_{}_mitigation_rl.pth".format(episode)
        save_path = os.path.join(self.config_dict["rl_checkpoint_savedir"], save_name)
        torch.save({
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'policy_net_target_state_dict': self.policy_net_target.state_dict(),
            'qvalue_net_state_dict': self.qvalue_net.state_dict(),
            'qvalue_net_target_state_dict': self.qvalue_net_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'rewards': rewards
        }, save_path)

    def load_checkpoint(self):
        raise NotImplementedError


def main():
    rl_lbc = CarlaLBCDDPG()
    # rl_lbc.inference_step_loop()
    # rl_lbc.inference_step_loop_threshold_mitigation()
    rl_lbc.training_step_loop_reinforcement_mitigation(500, 32, 32, 3.5, 50)


if __name__ == "__main__":
    main()
