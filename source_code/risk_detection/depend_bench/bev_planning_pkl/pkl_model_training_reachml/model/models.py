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
import torch
import torch.nn as nn
import torchvision.models as torch_models


class ReachNet(nn.Module):
    def __init__(self, num_input_channels, vstate_size, input_size, output_size):
        super().__init__()
        self.num_input_channels = num_input_channels + 1
        self.input_size = input_size
        self.output_size = output_size
        self.vstate_size = vstate_size
        self.downsampled_size = 4 * 16
        self.conv_output_channel = 512

        self.resnet = torch_models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(self.num_input_channels, 64,
                                      kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True)
        # get rid of the adaptive pooling and FC classification layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))
        # network prediction head
        self.linear_reduction = nn.Linear(self.downsampled_size * self.conv_output_channel,
                                          self.conv_output_channel // 128 * self.output_size[0] * self.output_size[1],
                                          bias=True)
        # self.output_linear1 = nn.Linear(self.conv_output_channel // 128 * self.output_size[0] * self.output_size[1],
        #                                 self.conv_output_channel // 128 * self.output_size[0] * self.output_size[1], bias=True)
        # self.output_linear2 = nn.Linear(self.conv_output_channel // 32 * self.output_size[0] * self.output_size[1],
        #                                 self.output_size[0] * self.output_size[1], bias=True)
        self.output_conv2 = nn.Conv2d(self.conv_output_channel // 128, 1,
                                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.output_sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # will not use vstate af first, so the following can change
        self.vstate_input_fc = nn.Linear(vstate_size, self.input_size[0] * self.input_size[1], bias=True)
        self.vstate_output_fc = nn.Linear(vstate_size, self.conv_output_channel // 32 * self.output_size[0] * self.output_size[1], bias=True)

    def forward(self, x1, x2):
        x2 = self.vstate_input_fc(x2)
        x2 = self.activation(x2)
        x2 = torch.reshape(x2, (-1, 1, self.input_size[0], self.input_size[1]))
        x1 = torch.cat([x1, x2], 1)
        x1 = self.resnet(x1)
        x1 = self.activation(x1)

        # flatten to batch x (channel x H x W)
        x1 = torch.flatten(x1, start_dim=1)
        x1 = self.linear_reduction(x1)
        x1 = self.activation(x1)
        x1 = self.dropout(x1)
        # x1 = self.output_linear1(x1)
        # x1 = self.activation(x1)
        # x1 = self.dropout(x1)

        # x2 = self.vstate_output_fc(x2)
        # x2 = self.activation(x2)
        # x2 = self.dropout(x2)
        # x1 = x1 + x2

        # x1 = self.output_linear2(x1)
        # x1 = torch.reshape(x1, (-1, 1, self.output_size[0], self.output_size[1]))

        x1 = torch.reshape(x1, (-1, self.conv_output_channel // 128, self.output_size[0], self.output_size[1]))
        # x1 = self.output_conv2(x1)
        # predicted_labels = self.output_sigmoid(x1)
        # return predicted_labels

        return self.output_conv2(x1)



