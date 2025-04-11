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

import pickle as pkl
import torch
import copy
import numpy as np
from reachml.model.inference import ReachInferencer
from reachml.model.reachml_utils import calculate_metrics
import matplotlib.pyplot as plt
import os

# model_path = "/media/sheng/data4/projects/DiverseEnv/carladataset/carla-sim/bev_planning_pkl/reachml/final_all_GT_ghost_cutin_2023-10-21-07-54-16.pth"
# model_path = "/media/sheng/data4/projects/DiverseEnv/carladataset/carla-sim/bev_planning_pkl/reachml/final_all_GT_wo_front_accident_2023-10-22-11-56-30.pth"
model_path = "/media/sheng/data4/projects/ReachML/reachml/checkpoints/RealEnv/final_pkl_single_frame_2023-12-01-23-40-41.pth"
config_path = "/media/sheng/data4/projects/RealEnv/argodataset/argoverse/generate_data_dyn/trail_fix_dyn_risk_analysis_data_config.json"
data_pkl = "/home/sheng/projects/ReachML/reachml/data/train_test_sim_lite/route_highway.xml_fi_ghost_cutin.json.pkl/model_unc/data_PS_GT_SINGLE_AGENT_fi_ghost_cutin_00216/6865256054_PS_GT_SINGLE_AGENT_fi_ghost_cutin_00216_data_generated.pkl"
data_pkl_folder = "/home/sheng/projects/ReachML/reachml/data/train_test_sim_lite/route_highway.xml_fi_ghost_cutin.json.pkl/model_unc/data_PS_GT_SINGLE_AGENT_fi_ghost_cutin_00216"
inference_driver = ReachInferencer(load_path=model_path, config_path=config_path)

def inference_folder(save_fig=False):
    for filename in os.listdir(data_pkl_folder):
        data_pkl = os.path.join(data_pkl_folder, filename)
        # checking if it is a file
        if "_00216_data_generated.pkl" in data_pkl:
            print("Inferencing", data_pkl)

            # load an example data
            with open(data_pkl, "rb") as f:
                data_dict = pkl.load(f)
                data_dict_ = data_dict[list(data_dict.keys())[0]]
            data_dict = data_dict_
            assert len(data_dict["featureMaps"]) == len(data_dict["egoState"])

            if not data_dict["featureMaps"]:
                print("Feature map not detected!")
                continue

            bev_features = torch.from_numpy(copy.deepcopy(data_dict["featureMaps"][0]))
            ego_states = torch.from_numpy(copy.deepcopy(data_dict["egoState"][0]))
            reach_labels = torch.from_numpy(copy.deepcopy(data_dict["reachableLabel"][0]))

            # add one dimension in front, bev_features: (1 x Channels x H x W), states: (1 x 5)
            bev_features = bev_features[None, :]
            ego_states = ego_states[None, :]
            reach_labels = reach_labels[None, :]
            predicted_labels = inference_driver.inference_wrapper(bev_features.detach().numpy(), ego_states.detach().numpy())

            # output (1 x 1 x #Grids vertically x #Grids horizontally)
            print(predicted_labels.shape)

            # sanity check
            predicted_labels_flat = torch.flatten(predicted_labels.cpu()).detach().numpy()
            local_labels_flat = torch.flatten(reach_labels.cpu()).detach().numpy()
            accuracy, precision, recall, f1_score = calculate_metrics(predicted_labels_flat, local_labels_flat)
            print("Accuracy: {}, Precision: {}, Recall: {}, F1-Score: {}".format(accuracy,
                                                                                 precision,
                                                                                 recall,
                                                                                 f1_score))


            predicted_labels_2d = np.reshape(predicted_labels_flat, (26, 17))
            if save_fig:
                fig, ax = plt.subplots()
                im = ax.imshow(predicted_labels_2d)
                # for i in range(26):
                #     for j in range(17):
                #         text = ax.text(j, i, predicted_labels_2d[i, j],
                #                        ha="center", va="center", color="w")
                ax.set_aspect(1/2)
                # plt.show()

                plt.savefig(data_pkl.split('_PS_GT_SINGLE_AGENT_fi_ghost_cutin_00216_data_generated.')[0]+".jpg", dpi=200)
                plt.close()

            return predicted_labels_2d


def inference_single_path(data_pkl=data_pkl, save_fig=False):
    if "_00216_data_generated.pkl" in data_pkl:
        print("Inferencing", data_pkl)

        # load an example data
        with open(data_pkl, "rb") as f:
            data_dict = pkl.load(f)
            data_dict_ = data_dict[list(data_dict.keys())[0]]
        data_dict = data_dict_
        assert len(data_dict["featureMaps"]) == len(data_dict["egoState"])

        if not data_dict["featureMaps"]:
            print("Feature map not detected!")
            return None

    bev_features = torch.from_numpy(copy.deepcopy(data_dict["featureMaps"][0]))
    ego_states = torch.from_numpy(copy.deepcopy(data_dict["egoState"][0]))
    reach_labels = torch.from_numpy(copy.deepcopy(data_dict["reachableLabel"][0]))

    # add one dimension in front, bev_features: (1 x Channels x H x W), states: (1 x 5)
    bev_features = bev_features[None, :]
    ego_states = ego_states[None, :]
    reach_labels = reach_labels[None, :]
    predicted_labels = inference_driver.inference_wrapper(bev_features.detach().numpy(), ego_states.detach().numpy())

    # output (1 x 1 x #Grids vertically x #Grids horizontally)
    print(predicted_labels.shape)

    # sanity check
    predicted_labels_flat = torch.flatten(predicted_labels.cpu()).detach().numpy()
    local_labels_flat = torch.flatten(reach_labels.cpu()).detach().numpy()
    accuracy, precision, recall, f1_score = calculate_metrics(predicted_labels_flat, local_labels_flat)
    print("Accuracy: {}, Precision: {}, Recall: {}, F1-Score: {}".format(accuracy,
                                                                         precision,
                                                                         recall,
                                                                         f1_score))

    predicted_labels_2d = np.reshape(predicted_labels_flat, (26, 17))
    if save_fig:
        fig, ax = plt.subplots()
        im = ax.imshow(predicted_labels_2d)
        ax.set_aspect(1/2)

        plt.savefig(data_pkl.split('_PS_GT_SINGLE_AGENT_fi_ghost_cutin_00216_data_generated.')[0]+".jpg", dpi=200)
        plt.close()

    return predicted_labels_2d



def inference_single(bev_features, ego_states, save_fig=False):
    # add one dimension in front, bev_features: (1 x Channels x H x W), states: (1 x 5)

    predicted_labels = inference_driver.inference_wrapper(np.array(bev_features), np.array(ego_states))

    # output (1 x 1 x #Grids vertically x #Grids horizontally)
    # print(predicted_labels.shape)

    # sanity check
    predicted_labels_flat = torch.flatten(predicted_labels.cpu()).float().detach().numpy()
    predicted_labels_2d = np.reshape(predicted_labels_flat, (13, 13))

    if save_fig:
        fig, ax = plt.subplots()
        im = ax.imshow(predicted_labels_2d)
        ax.set_aspect(1/2)

        plt.savefig(data_pkl.split('_PS_GT_SINGLE_AGENT_fi_ghost_cutin_00216_data_generated.')[0]+".jpg", dpi=200)
        plt.close()

    return predicted_labels_2d

# example inferencer usage
if __name__ == "__main__":
    pass




