import numpy as np
import torch

from torch import nn
from .models import SegmentationModelQValue
from .models import SegmentationModel
from .utils.heatmap import ToHeatmap
from .converter import Converter


class QvalueModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.to_heatmap = ToHeatmap(self.hparams["heatmap_radius"])
        self.net = SegmentationModelQValue(16, 1, hack=self.hparams["hack"], temperature=self.hparams["temperature"], output_dim=hparams["action_space"])
        self.relu = nn.ReLU()

    def forward(self, img, target):
        """
        input: img state, target, action
        output: Q value
        """
        # target_heatmap_cam = self.to_heatmap(target, img)[:, None]
        target_heatmap_cam = list()
        for k in range(target.shape[1]):
            curr_target = target[:, k, :]
            target_heatmap_cam.append(self.to_heatmap(curr_target, img))
        target_heatmap_cam = torch.stack(target_heatmap_cam, dim=1)
        nn_input = torch.cat((img, target_heatmap_cam), 1)
        q_values = self.net(nn_input)
        return q_values


class ActorModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.to_heatmap = ToHeatmap(self.hparams["heatmap_radius"])
        self.net = SegmentationModel(10, 4, hack=self.hparams["hack"], temperature=self.hparams["temperature"])
        self.converter = Converter()
        self.eval()

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
