import numpy as np
import torch

from torch import nn
from .models import SegmentationModelQValueDuel
from .utils.heatmap import ToHeatmap
from .converter import Converter


class DuelQvalueModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.to_heatmap = ToHeatmap(self.hparams["heatmap_radius"])
        self.net = SegmentationModelQValueDuel(
            16, 1, self.hparams["action_space"], hack=self.hparams["hack"], temperature=self.hparams["temperature"], output_dim=hparams["action_space"]
        )
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
        duel_value = self.net(nn_input)
        return duel_value
