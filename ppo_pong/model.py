import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Weight initialization
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, num_actions, config):
        super(ActorCritic, self).__init__()
        
        # Extracts features from the stacked frames
        self.cnn_base = nn.Sequential(
            layer_init(nn.Conv2d(config["FRAME_STACK_SIZE"], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        # Determine the output size of the CNN base
        with torch.no_grad():
            dummy_input = torch.zeros(1, config["FRAME_STACK_SIZE"], *config["IMAGE_SIZE"])
            cnn_out_dim = self.cnn_base(dummy_input).shape[1]

        # Takes the features from the CNN and decides on an action
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, num_actions), std=0.01)
        )

        # Takes the features from the CNN and estimates the value of the state
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0)
        )

    # Takes a batch of observations and returns action logits and state values
    def forward(self, x):
        cnn_features = self.cnn_base(x)
        action_logits = self.actor_head(cnn_features)
        value_estimate = self.critic_head(cnn_features)
        return action_logits, value_estimate

    # Runs the forward pass, creates a probability distribution, samples an action
    # and computes the log probability and entropy.
    def get_action_and_value(self, x, action=None):
        action_logits, value = self(x)
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value