import numpy as np
import torch
from torch import nn


from .models import SegmentationModelQValue
from .models import SegmentationModel
from .utils.heatmap import ToHeatmap
from .dataset import get_dataset
from .converter import Converter
from .scripts.cluster_points import points as RANDOM_POINTS


class CriticModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.to_heatmap = ToHeatmap(self.hparams["heatmap_radius"])
        self.net = SegmentationModelQValue(11, 1, hack=self.hparams["hack"], temperature=self.hparams["temperature"])
        self.relu = nn.ReLU()
        self.action_linear = nn.Linear(8, 9216)
        self.action_deconv = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1)

    def forward(self, img, target, action):
        """
        input: img state, target, action
        output: Q value
        """
        modified_action = torch.flatten(action, start_dim=1)
        modified_action = self.action_linear(modified_action)
        modified_action = self.relu(modified_action)
        modified_action = torch.reshape(modified_action, (-1, 1, 72, 128))
        modified_action = self.action_deconv(modified_action, output_size=(1,1,144,256))
        modified_action = self.relu(modified_action)
        target_heatmap_cam = self.to_heatmap(target, img)[:, None]
        nn_input = torch.cat((img, target_heatmap_cam, modified_action), 1)
        q_values = self.net(nn_input)
        return q_values

class PolicyModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.to_heatmap = ToHeatmap(self.hparams["heatmap_radius"])
        self.net = SegmentationModel(10, 4, hack=self.hparams["hack"], temperature=self.hparams["temperature"])
        self.converter = Converter()

    def forward(self, img, target, debug=False):
        """
        input: img state, target
        output: Action points
        """
        # target_cam = self.converter.map_to_cam(target)
        target_heatmap_cam = self.to_heatmap(target, img)[:, None]
        nn_input = torch.cat((img, target_heatmap_cam), 1)
        out = self.net(nn_input)
        if debug:
            print(img)
            print(target_heatmap_cam)
            print(out)
        return out, (torch.from_numpy(np.array([])), torch.from_numpy(np.array([])))
