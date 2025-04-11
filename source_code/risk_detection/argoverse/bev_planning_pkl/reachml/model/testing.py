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
import time
import numpy as np
from torch.utils import data
from utils import calculate_metrics
from models import ReachNet
from dataset import ReachabilityDataset

params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 1}

device = torch.device("cuda:0")


class Tester:
    def __init__(self, data_path, load_path, config_path):
        self.model = None
        self.data_path = data_path
        self.load_path = load_path
        self.config_path = config_path
        self.testing_set = None
        self.testing_generator = None
        self.criterion = nn.BCELoss(reduction='sum')
        self.prepare_data()
        self.init_model()
        print("Done initialization...")

    def init_model(self):
        with open(self.config_path, "r") as f:
            config = json.load(f)
        self.model = ReachNet(int(config["numInChannels"]),
                              int(config["vStateSize"]),
                              config['inputSize'],
                              config['outputSize'])
        self.model.to(device)

    def prepare_data(self):
        file_list = list()
        with open(self.data_path, "r") as f:
            for line in f:
                file_list.append(line.strip())
        testing = list()
        for f_name in file_list:
            testing.append(f_name)
        partition = {"test": testing}
        self.testing_set = ReachabilityDataset(partition["test"])
        self.testing_generator = data.DataLoader(self.testing_set, **params)

    def test(self):
        if not self.load_path:
            raise FileNotFoundError
        self.model.load_state_dict(torch.load(self.load_path))
        print("Start inference...")
        testing_loss_hist = list()
        cum_predicted_labels = list()
        cum_local_labels = list()
        test_start_time = time.time()
        self.model.eval()  # turn on evaluation mode
        for local_batch, local_states, local_labels in self.testing_generator:
            local_batch = local_batch.float().to(device)
            local_states = local_states.float().to(device)
            local_labels = local_labels.float().to(device)
            predicted_labels = self.model(local_batch, local_states)
            loss = self.criterion(predicted_labels, local_labels)
            print("Sum batch testing loss:", loss.item())
            testing_loss_hist.append(loss.cpu().detach().item())
            predicted_labels_flat = list(torch.flatten(predicted_labels.cpu()).detach().numpy())
            local_labels_flat = list(torch.flatten(local_labels.cpu()).detach().numpy())
            cum_predicted_labels.extend(predicted_labels_flat)
            cum_local_labels.extend(local_labels_flat)
        test_end_time = time.time()
        test_data_size = len(self.testing_generator.dataset)
        print("Inference latency {}ms per datapoint.".format((test_end_time - test_start_time) * 1000 / test_data_size))
        average_testing_loss = sum(testing_loss_hist) / test_data_size
        print("Avg. Testing Loss: {}.".format(average_testing_loss))
        cum_predicted_labels = np.array(cum_predicted_labels)
        cum_local_labels = np.array(cum_local_labels)
        accuracy, precision, recall, f1_score = calculate_metrics(cum_predicted_labels, cum_local_labels)
        print("Accuracy: {}, Precision: {}, Recall: {}, F1-Score: {}".format(accuracy,
                                                                             precision,
                                                                             recall,
                                                                             f1_score))
