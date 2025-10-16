import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import torch
import numpy as np
import math
from agent import DQNAgent
from gymnasium.wrappers import AtariPreprocessing, FrameStack, RecordEpisodeStatistics
from collections import deque
import time
import random
import wandb
from utils import record_and_save_gif, make_video_env
import os

from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

# Hyperparameters and setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

config = {
    "LEARNING_RATE": 0.00025,     # Step size taken to minimize loss function
    "MAX_STEPS": 2_500_000,       # Number of agent-environment interactions
    "BATCH_SIZE": 32,             # Number of transitions to sample from the replay buffer
    "GAMMA": 0.99,                # Discount factor for future rewards
    "EPS_START": 1.0,             # Initial value of epsilon (100% random actions)
    "EPS_END": 0.01,              # Final value of epsilon (1% random actions)
    "EPS_DECAY": 1_000_000,       # How many steps to decay epsilon over
    "BUFFER_SIZE": 50_000,        # Store experiences
    "TARGET_UPDATE_FREQ": 10_000, # How often to copy weights from policy_net to target_net
    "LOG_INTERVAL_STEPS": 10_000  # How often to log progress
}

# Experiment tracking and visualization
wandb.init(
    project="dqn-atari-breakout",
    config=config
)


# Environment Setup
env = gym.make("BreakoutNoFrameskip-v4")
env = MaxAndSkipEnv(env, skip=4)
env = AtariPreprocessing(env, frame_skip=1, screen_size=84, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=False)
env = FrameStack(env, 4)
env = RecordEpisodeStatistics(env)
n_actions = env.action_space.n
input_shape = (4, 84, 84) # 4 stacked frames, size 84x84

# Agent initialization
agent = DQNAgent(
    n_actions=n_actions,
    input_shape=input_shape,
    device=device,
    lr=config["LEARNING_RATE"],
    gamma=config["GAMMA"],
    buffer_size=config["BUFFER_SIZE"]
)
total_steps = 0
i_episode = 0
game_record = 0
state, info = env.reset()
state = np.array(state)

while total_steps < config["MAX_STEPS"]:
    
    # Calculate epsilon and select action
    epsilon = config["EPS_END"] + (config["EPS_START"] - config["EPS_END"]) * \
              math.exp(-1. * total_steps / config["EPS_DECAY"])
    action = agent.act(state, epsilon)

    # Take a step in the environment
    next_state, reward, terminated, truncated, info = env.step(action)
    next_state = np.array(next_state)
    done = terminated or truncated

    clipped_reward = np.sign(reward) # Stabilize training

    agent.remember(state, action, clipped_reward, next_state, done) # Store the transition in the replay buffer
    
    state = next_state

    loss = agent.replay(config["BATCH_SIZE"])
    total_steps += 1

    # Update target network
    if total_steps % config["TARGET_UPDATE_FREQ"] == 0:
        agent.update_target_net()
    
    # Log progress
    if total_steps % config["LOG_INTERVAL_STEPS"] == 0:
        loss_str = f"{loss:.5f}" if loss is not None else "N/A"

        if len(env.return_queue) > 0:
            mean_score = np.mean(env.return_queue)
            current_max_score = np.max(env.return_queue).item()
            game_record = max(game_record, current_max_score)
        else:
            mean_score = 0.0

        print(f"Steps: {total_steps} ({ (total_steps/config['MAX_STEPS'])*100 :.2f}%), Episode: {i_episode}, Mean score(100): {mean_score:.2f}, Record: {game_record} Epsilon: {epsilon:.4f}, Loss: {loss_str}")

        wandb.log({
            "charts/mean_score_100_ep": mean_score,
            "charts/record_score": game_record,
            "charts/epsilon": epsilon,
            "losses/loss": loss,
            "diagnostics/episode": i_episode,
            "diagnostics/buffer_size": len(agent.memory),
        }, step=total_steps)

    if done:
        if 'episode' in info:
            episodic_return = info['episode']['r'].item()
            wandb.log({"charts/episodic_return": episodic_return}, step=total_steps)
            
            if episodic_return > game_record:
                game_record = episodic_return
                print(f"New record {game_record}, saving model and recording GIF...")
                
                # Save model
                model_path = "best_model.pth"
                torch.save(agent.policy_net.state_dict(), model_path)

                # Make folder for GIF
                gif_dir = "training_gifs"
                os.makedirs(gif_dir, exist_ok=True)
                
                video_env = make_video_env(config)
                gif_path = os.path.join(gif_dir, f"record_score_{game_record:.0f}_step_{total_steps}.gif")
                record_and_save_gif(agent, video_env, device, gif_path)
                video_env.close()

        state, info = env.reset()
        state = np.array(state)
        i_episode += 1
print("Training ended")
env.close()
wandb.finish()