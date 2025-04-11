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
import os
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from reachml.model.utils import calculate_metrics
from datetime import datetime
from torch.utils import data
from models import ReachNet
from dataset import ReachabilityDataset

params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 32}

val_params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 32}

device = torch.device("cuda:0")


class Trainer:
    def __init__(self, data_path, load_path, save_path, config_path):
        self.model = None
        self.data_path = data_path
        self.save_path = save_path
        self.load_path = load_path
        self.config_path = config_path
        self.training_set = None
        self.training_generator = None
        self.validation_set = None
        self.validation_generator = None
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.optimizer = None
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

    def prepare_data(self, partition=0.8):
        file_list = list()
        with open(self.data_path, "r") as f:
            for line in f:
                file_list.append(line.strip())
        training = list()
        validation = list()
        for f_name in file_list:
            if np.random.rand() <= partition:
                training.append(f_name)
            else:
                validation.append(f_name)
        partition = {"train": training, "validate": validation}
        self.training_set = ReachabilityDataset(partition["train"])
        self.training_generator = data.DataLoader(self.training_set, **params)
        self.validation_set = ReachabilityDataset(partition["validate"])
        self.validation_generator = data.DataLoader(self.validation_set, **val_params)

    def train(self, lr, epochs, save_weight_prefix):
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d-%H-%M-%S")
        print("Starts training...")
        if self.load_path:
            raise NotImplementedError
        self.optimizer = optim.Adam(self.model.parameters(), lr, weight_decay=0.99)
        training_loss_hist_total = list()
        validation_loss_hist_total = list()
        validation_accuracy_hist_total = list()
        validation_precision_hist_total = list()
        validation_recall_hist_total = list()
        validation_f1_score_hist_total = list()
        for epoch in tqdm(range(epochs)):
            training_loss_hist = list()
            validation_loss_hist = list()
            self.model.train()
            for local_batch, local_states, local_labels in self.training_generator:
                local_batch = local_batch.float().to(device)
                local_states = local_states.float().to(device)
                local_labels = local_labels.float().to(device)
                predicted_labels = self.model(local_batch, local_states)
                loss = self.criterion(predicted_labels, local_labels)
                training_loss_hist.append(loss.cpu().detach().item())
                self.optimizer.zero_grad()
                loss.backward()
                print("Sum batch training loss:", loss.item())
                self.optimizer.step()

            # validation after training for each epoch
            self.model.eval()
            val_start_time = time.time()
            cum_predicted_labels = list()
            cum_local_labels = list()
            for local_batch, local_states, local_labels in self.validation_generator:
                local_batch = local_batch.float().to(device)
                local_states = local_states.float().to(device)
                local_labels = local_labels.float().to(device)
                predicted_labels = self.model(local_batch, local_states)
                loss = self.criterion(predicted_labels, local_labels)
                print("Sum batch validation loss:", loss.item())
                validation_loss_hist.append(loss.cpu().detach().item())
                predicted_labels_flat = list(torch.flatten(predicted_labels.cpu()).detach().numpy())
                local_labels_flat = list(torch.flatten(local_labels.cpu()).detach().numpy())
                cum_predicted_labels.extend(predicted_labels_flat)
                cum_local_labels.extend(local_labels_flat)
            val_end_time = time.time()
            val_data_size = len(self.validation_generator.dataset)
            train_data_size = len(self.training_generator.dataset)
            print("Inference latency {}ms per datapoint.".format((val_end_time - val_start_time) * 1000 / val_data_size))
            average_training_loss = sum(training_loss_hist) / train_data_size
            average_validation_loss = sum(validation_loss_hist) / val_data_size
            print("Epoch: {}, Avg. Training Loss: {}, Avg. Validation Loss: {}".format(epoch, average_training_loss,
                                                                                       average_validation_loss))
            cum_predicted_labels = np.array(cum_predicted_labels)
            cum_local_labels = np.array(cum_local_labels)
            accuracy, precision, recall, f1_score = calculate_metrics(cum_predicted_labels, cum_local_labels)
            print("Accuracy: {}, Precision: {}, Recall: {}, F1-Score: {}".format(accuracy,
                                                                                 precision,
                                                                                 recall,
                                                                                 f1_score))
            training_loss_hist_total.append(average_training_loss)
            validation_loss_hist_total.append(average_validation_loss)
            validation_accuracy_hist_total.append(accuracy)
            validation_precision_hist_total.append(precision)
            validation_recall_hist_total.append(recall)
            validation_f1_score_hist_total.append(f1_score)

        # done training, save model weights and plot losses
        weight_name = "{}_{}.pth".format(save_weight_prefix, ts)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, weight_name))

        epoch_list = [i for i in range(epochs)]
        plt.figure()
        plt.plot(epoch_list, training_loss_hist_total, label="Train")
        plt.plot(epoch_list, validation_loss_hist_total, label="Val")
        plt.title("Loss Overtime")
        plt.xlabel("Epoch")
        plt.ylabel("MSELoss")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(epoch_list, validation_accuracy_hist_total, label="Accuracy")
        plt.plot(epoch_list, validation_precision_hist_total, label="Precision")
        plt.plot(epoch_list, validation_recall_hist_total, label="Recall")
        plt.plot(epoch_list, validation_f1_score_hist_total, label="F1-Score")
        plt.title("Performance Overtime")
        plt.xlabel("Epoch")
        plt.ylabel("")
        plt.legend()
        plt.show()
