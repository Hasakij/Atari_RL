import gymnasium as gym
import torch
import imageio
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

# Create a single Atari environment with specified wrappers
def make_video_env(config):
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    env = MaxAndSkipEnv(env, skip=4)
    env = AtariPreprocessing(env, frame_skip=1, screen_size=84, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=False)
    env = FrameStack(env, 4)
    return env

def record_and_save_gif(agent, env, device, gif_path):
    frames = []
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    print("Recording GIF...")
    while not done:
        # Render the current frame from the environment
        frame = env.render()
        frames.append(frame)
        
        # Agent acts deterministically (epsilon=0)
        action = agent.act(state, epsilon=0.0)
        
        # Take a step in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        state = next_state
        total_reward += reward
    
    # Add the final frame
    frames.append(env.render())
    
    if frames:
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Saved GIF of full episode: {gif_path}, Total Reward: {total_reward:.2f}")
    
    return total_reward