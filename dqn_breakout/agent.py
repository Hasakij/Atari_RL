import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from model import ConvDQN 

# Define DQN Agent with Experience Replay Buffer
class DQNAgent:
    def __init__(self, n_actions, input_shape, device, lr, gamma, buffer_size):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma                                           # Discount factor for future rewards
        self.memory = deque(maxlen=buffer_size)                      # Store (state, action, reward, next_state, done) tuples
        self.policy_net = ConvDQN(input_shape, n_actions).to(device) # Select actions and compute the Q-values for the current state
        self.target_net = ConvDQN(input_shape, n_actions).to(device) # Copy of the polcy net with frozen weights, calculate the value of the next state
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)

    def act(self, state, epsilon):

        # Selects an action using an epsilon-greedy policy
        if np.random.rand() <= epsilon:
            return random.randrange(self.n_actions)

        # Convert state to a tensor, add a batch dimension,
        # pass it to the policy network
        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(state_tensor)

            # Selects an action with the highest Q-value
            return q_values.argmax().item()

    # Experience replay buffer: stores a transition
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Copies the weights from the policy network to the target network
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # Samples a minibatch from the experience replay buffer and performs a Q-learning update.
    def replay(self, batch_size):

        # Don't update if buffer not large enough 
        if len(self.memory) < batch_size:
            return None

        # Sample a random minibatch of transitions
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors
        state_batch = torch.tensor(np.array(states), device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(actions, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        # Create a mask to identify non-terminal states (episodes that have not ended)
        non_final_mask = torch.tensor(tuple(map(lambda d: not d, dones)), device=self.device, dtype=torch.bool)
        non_final_next_states_list = [s for s, d in zip(next_states, dones) if not d]

        # Compute Q(s_t, a_t) for the current states
        state_action_values = self.policy_net(state_batch).gather(1, action_batch) # "gather" to select the Q-values corresponding to the actions that were actually taken.

        # Initialize the value of the next states to zero for all transitions
        next_state_values = torch.zeros(batch_size, device=self.device)

        # For non-terminal states, compute the value V(s_{t+1}) using the target network
        if len(non_final_next_states_list) > 0:
            non_final_next_states = torch.tensor(np.array(non_final_next_states_list), device=self.device, dtype=torch.float32)
            with torch.no_grad():

                # Double DQN
                # Use policy network to take best action for next state
                best_actions = self.policy_net(non_final_next_states).argmax(1).unsqueeze(1)

                # Use target network to evaluate Q-value of this action
                q_values_from_target = self.target_net(non_final_next_states)
                next_state_values[non_final_mask] = q_values_from_target.gather(1, best_actions).squeeze(1)

        # Calculate expected Q-values (the learning target): y_t = r_t + gamma * V(s_{t+1})
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Calculate loss between current and expected Q-values 
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()

        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()