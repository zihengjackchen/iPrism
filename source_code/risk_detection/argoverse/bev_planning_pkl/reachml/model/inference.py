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
import torch
import torch.nn as nn
from reachml.model.models import ReachNet

params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 1}

# device = torch.device("cpu")
device = torch.device("cuda:0")


class ReachInferencer:
    def __init__(self, load_path, config_path):
        self.model = None
        self.load_path = load_path
        self.config_path = config_path
        self.testing_set = None
        self.testing_generator = None
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.init_model()
        print("Done ReachNet initialization...")

    def init_model(self):
        with open(self.config_path, "r") as f:
            config = json.load(f)
        self.model = ReachNet(int(config["numInChannels"]),
                              int(config["vStateSize"]),
                              config['inputSize'],
                              config['outputSize'])
        # self.model.load_state_dict(torch.load(self.load_path, map_location=torch.device('cpu')))
        self.model.load_state_dict(torch.load(self.load_path))
        self.model = self.model.half()
        self.model.to(device)

    def inference_wrapper(self, local_batch, local_states):
        with torch.no_grad():
            self.model.eval()
            local_batch = torch.from_numpy(local_batch)
            local_states = torch.from_numpy(local_states)
            local_batch = local_batch.float().to(device)
            local_states = local_states.float().to(device)
            predicted_labels = self.model(local_batch.half(), local_states.half())
        return predicted_labels
