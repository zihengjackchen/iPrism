def training_step_loop_reinforcement_mitigation(
        self, episodes, batch_size, start_training_replaybuffer_size, skip_first_seconds, skip_mitigation_seconds,
        save_frequency):
    assert start_training_replaybuffer_size >= batch_size
    self.all_rewards = []
    self.avg_rewards = []
    self.total_frame_counter = 0
    for episode in range(episodes):
        self.start_sim()
        episode_reward = 0
        self.config_dict["trafficManagerPort"] = 8000 + episode
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
        if PLOT:
            plt.figure(figsize=(5, 2.5))
        while not done:
            # get data and then apply action
            input_data, timestamp, rewards = self.carla_rl_env.step_sense()
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
                self.total_frame_counter += 1
                if timestamp.elapsed_seconds > skip_first_seconds:
                    print("At {} seconds, saving state-action tuples.".format(timestamp.elapsed_seconds))
                    assert len(prev_state_list) >= 3
                    current_prev_three_states = [prev_state_list[-3], prev_state_list[-2], prev_state_list[-1],
                                                    [img, target]]
                    state_imgs = [item[0].squeeze() for item in current_prev_three_states]
                    state_imgs = torch.vstack(state_imgs)
                    state_targets = [item[1].squeeze() for item in current_prev_three_states]
                    state_targets = torch.vstack(state_targets)
                    states = [state_imgs, state_targets]
                    memory_blob = [
                        copy.deepcopy([prev_states[0].cpu(), prev_states[1].cpu().float()]),
                        prev_action,
                        copy.deepcopy(rewards),
                        copy.deepcopy([states[0].cpu(), states[1].cpu().float()]), copy.deepcopy(int(done))
                    ]
                    if not np.isnan(rewards[1]) and prev_states is not None and not len(prev_states[0]) == 1:
                        self.replay_buffer.add(memory_blob)

                    # calculate episode_reward
                    if not np.isnan(rewards[1]):
                        episode_reward += rewards[1]
                    else:
                        episode_reward += 0
                else:
                    print("Skip save, the game just started.")

                # act on current state
                points, (target_cam, _) = self.policy_net(img, target)
                mitigation_action = 0
                if timestamp.elapsed_seconds > skip_mitigation_seconds and len(
                        self.replay_buffer) > start_training_replaybuffer_size:
                    epsilon = self.epsilon_by_frame(self.total_update_steps)
                    mitigation_action = self.get_action(states, epsilon)
                    print("Mitigation action:", mitigation_action)
                    points = self.get_mitigation_action(points, mitigation_action)

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
                    print("!!!!!!!!!!!!!!!!!! Update !!!!!!!!!!!!!!!!!!")
                    self.update_agent(batch_size)
            else:
                # collect previous state information
                states = [img, target]
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
            print("Total reward:", episode_reward)

            if done:
                # write the final memory
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
        if PLOT:
            plt.close()
        self.carla_rl_env.post_step()
        del self.carla_rl_env
        print("================= Finish episode {} =================".format(episode))
        self.all_rewards.append(episode_reward)
        self.avg_rewards.append(np.mean(self.all_rewards[-10:]))
        print(
            "Episode: {}, reward: {}, average_reward: {} \n".format(episode, np.round(episode_reward, decimals=2),
                                                                    np.mean(self.all_rewards[-10:]))
        )
        # save weight
        if episode % save_frequency == 0:
            self.save_checkpoint(episode, self.all_rewards)
        writer.add_scalar("Episode rewards", episode_reward, episode)
        writer.flush()
        self.stop_sim()

    # all done
    print("Done training {} episodes.".format(episodes))
    self.save_checkpoint("final", self.all_rewards)
    writer.close()