from collections import deque
import random
import copy


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)

    def add(self, experience):
        self.memory.appendleft(copy.deepcopy(experience))

    def random_draw(self, sample_size):
        state_batch = list()
        action_batch = list()
        future_state_batch = list()
        reward_batch = list()
        terminal_batch = list()
        batch = random.sample(self.memory, sample_size)
        for state, action, future_state, reward, terminal in batch:
            state_batch.append(state)
            action_batch.append(action)
            future_state_batch.append(future_state)
            reward_batch.append(reward)
            terminal_batch.append(terminal)
        return state_batch, action_batch, future_state_batch, reward_batch, terminal_batch

    def __len__(self):
        return len(self.memory)
