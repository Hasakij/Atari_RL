import torch
import torch.nn as nn
import numpy as np

# Deep Q-Network (DQN) Model
# Network takes stacked frames from the Atari environment as input and outputs
# the estimated Q-value for each possible action
class ConvDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ConvDQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), # input_shape[0] - number of stacked frames 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # The fully connected layers take the flattened features from the convolutional
        # layers and map them to the Q-values for each action.
        self.fc_layers = nn.Sequential(
            nn.Linear(self.calc_conv_output(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    # Calculate the flattened output size of the convolutional layers
    def calc_conv_output(self, shape):
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.size()))

    # forward pass of the network, takes a batch of observations,
    # normalizes them, and returns Q-values.
    def forward(self, x):
        conv_out = self.conv_layers(x / 255.0).view(x.size()[0], -1)
        return self.fc_layers(conv_out)