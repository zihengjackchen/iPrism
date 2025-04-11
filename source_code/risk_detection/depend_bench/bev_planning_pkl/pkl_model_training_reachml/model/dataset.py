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
import copy
import pickle as pkl
import torch
from torch.utils import data


class ReachabilityDataset(data.Dataset):
    def __init__(self, pkl_list):
        self.reach_labels = list()
        self.ego_states = list()
        self.bev_features = list()
        self.pkl_list = pkl_list
        for f_pkl in pkl_list:
            with open(f_pkl, "rb") as f:
                try:
                    data_dict = pkl.load(f)
                except Exception as e:
                    print(e)
                    print("Skipping:", f_pkl)
                    continue
                data_dict_ = data_dict[list(data_dict.keys())[0]]
                if len(data_dict_["reachableLabel"]) == 0 or len(data_dict_["egoState"]) == 0 or len(data_dict_["featureMaps"]) == 0:
                    continue
                data_dict = data_dict_
                assert len(data_dict["reachableLabel"]) == len(data_dict["egoState"])
                assert len(data_dict["featureMaps"]) == len(data_dict["egoState"])
                for idx in range(len(data_dict["reachableLabel"])):
                    self.reach_labels.append(torch.from_numpy(copy.deepcopy(data_dict["reachableLabel"][idx])))
                    self.ego_states.append(torch.from_numpy(copy.deepcopy(data_dict["egoState"][idx])))
                    self.bev_features.append(torch.from_numpy(copy.deepcopy(data_dict["featureMaps"][idx])))
            print("Loaded data file: {}".format(f_pkl))
        assert len(self.reach_labels) == len(self.ego_states)
        assert len(self.ego_states) == len(self.bev_features)
        self.data_size = len(self.bev_features)
        print("Data size:", self.data_size)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.bev_features[index], self.ego_states[index], self.reach_labels[index]

