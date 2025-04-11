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
import gc
import json
import os
import shutil
import time
import random
import torch
import glob
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
          'num_workers': 1}

val_params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1}

device = torch.device("cuda:0")

torch.multiprocessing.set_sharing_strategy('file_system')


class TrainerLargeScale:
    def __init__(
            self, data_path, load_path, save_path,
            config_path, data_update_frequency, number_of_data,
            weight_save_frequency
    ):
        """
        data_update_frequency: number of episode per data reload
        number_of_data: number of datapoints to load at each reload
        """
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
        self.data_update_frequency = data_update_frequency
        self.sampling_size = number_of_data
        self.init_model()
        self.file_list = list()
        self.current_extract_dir = list()
        self.weight_save_frequency = weight_save_frequency
        with open(self.data_path, "r") as f:
            for line in f:
                self.file_list.append(line.strip())
        print("Done initialization...")

    def init_model(self):
        with open(self.config_path, "r") as f:
            config = json.load(f)
        self.model = ReachNet(int(config["numInChannels"]),
                              int(config["vStateSize"]),
                              config['inputSize'],
                              config['outputSize'])
        self.model.to(device)

    def load_data_from_disk(self, partition=0.8):
        episode_file_list = random.sample(self.file_list, self.sampling_size)

        # need to unzip the files first
        pkl_files_list = list()
        for f_name in episode_file_list:
            assert ".zip" in f_name
            extract_dir = f_name.split(".zip")[0]
            try:
                print("Unpacking {} to {}.".format(f_name, extract_dir))
                shutil.unpack_archive(f_name, extract_dir, "zip")
                print("Unpacked {} to {}.".format(f_name, extract_dir))
            except Exception as e:
                print(e)
                if os.path.isdir(extract_dir):
                    print("Removing: {} due to exception.".format(extract_dir))
                    shutil.rmtree(extract_dir)
                continue
            pkl_files = glob.glob(os.path.join(extract_dir, '*.pkl'), recursive=True)
            if len(pkl_files) == 0:
                pkl_files = glob.glob(os.path.join(extract_dir, '**/*.pkl'), recursive=True)
            pkl_files_list.extend(pkl_files)
            self.current_extract_dir.append(extract_dir)
        print("Done unpacking:", self.current_extract_dir)
        # then read the pkl files
        training = list()
        validation = list()
        for f_name in pkl_files_list:
            assert ".pkl" in f_name
            if np.random.rand() <= partition:
                training.append(f_name)
            else:
                validation.append(f_name)
        partition = {"train": training, "validate": validation}
        self.training_set = ReachabilityDataset(partition["train"])
        self.training_generator = data.DataLoader(self.training_set, **params)
        self.validation_set = ReachabilityDataset(partition["validate"])
        self.validation_generator = data.DataLoader(self.validation_set, **val_params)

    def clear_data_residual(self):
        for dir in self.current_extract_dir:
            if os.path.isdir(dir):
                print("Removing: {} after training.".format(dir))
                shutil.rmtree(dir)
            done_file = dir + ".done"
            print("Writing done file: {}".format(done_file))
            f = open(done_file, "w")
            f.close()
        self.current_extract_dir = list()
        self.training_set = None
        del self.training_set
        self.training_generator = None
        del self.training_generator
        self.validation_set = None
        del self.validation_set
        self.validation_generator = None
        del self.validation_generator
        gc.collect()

    def train(self, lr, epochs, save_weight_prefix):
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d-%H-%M-%S")
        print("Starts training...")
        if self.load_path:
            self.model.load_state_dict(torch.load(self.load_path))
        self.optimizer = optim.Adam(self.model.parameters(), lr, weight_decay=0.99)
        training_loss_hist_total = list()
        validation_loss_hist_total = list()
        validation_accuracy_hist_total = list()
        validation_precision_hist_total = list()
        validation_recall_hist_total = list()
        validation_f1_score_hist_total = list()
        for epoch in tqdm(range(epochs)):
            if epoch % self.data_update_frequency == 0:
                self.clear_data_residual()
                self.load_data_from_disk()
            training_loss_hist = list()
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
            train_data_size = len(self.training_generator.dataset)
            average_training_loss = sum(training_loss_hist) / train_data_size
            print("Epoch: {}, Avg. Training Loss: {}".format(epoch, average_training_loss))
            training_loss_hist_total.append(average_training_loss)

            if (epoch + 1) % self.data_update_frequency == 0:
                # validation after training for certain epoch
                validation_loss_hist = list()
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

                print(
                    "Inference latency {}ms per datapoint.".format((val_end_time - val_start_time) * 1000 / val_data_size))
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
                validation_loss_hist_total.append(average_validation_loss)
                validation_accuracy_hist_total.append(accuracy)
                validation_precision_hist_total.append(precision)
                validation_recall_hist_total.append(recall)
                validation_f1_score_hist_total.append(f1_score)

            if epoch % self.weight_save_frequency == 0:
                weight_name = "{}_{}_{}.pth".format(str(epoch).zfill(5), save_weight_prefix, ts)
                torch.save(self.model.state_dict(), os.path.join(self.save_path, weight_name))

        self.clear_data_residual()

        # done training, save model weights and plot losses
        weight_name = "final_{}_{}.pth".format(save_weight_prefix, ts)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, weight_name))

        plt.figure()
        plt.plot([i for i in range(len(training_loss_hist_total))], training_loss_hist_total, label="Train")
        plt.plot([i for i in range(len(validation_loss_hist_total))], validation_loss_hist_total, label="Val")
        plt.title("Loss Overtime")
        plt.xlabel("Epoch")
        plt.ylabel("Training MSELoss")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot([i for i in range(len(validation_loss_hist_total))], validation_loss_hist_total, label="Val")
        plt.title("Loss Overtime")
        plt.xlabel("Epoch")
        plt.ylabel("Testing MSELoss")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot([i for i in range(len(validation_accuracy_hist_total))], validation_accuracy_hist_total, label="Accuracy")
        plt.plot([i for i in range(len(validation_precision_hist_total))], validation_precision_hist_total, label="Precision")
        plt.plot([i for i in range(len(validation_recall_hist_total))], validation_recall_hist_total, label="Recall")
        plt.plot([i for i in range(len(validation_f1_score_hist_total))], validation_f1_score_hist_total, label="F1-Score")
        plt.title("Performance Overtime")
        plt.xlabel("Epoch")
        plt.ylabel("")
        plt.legend()
        plt.show()
