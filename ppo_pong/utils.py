import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import torch
import imageio
import numpy as np

# Create a single Atari environment with specified wrappers
def make_env(config, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(config["ENV_NAME"], frameskip=1, render_mode="rgb_array", repeat_action_probability=0.0)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda e: e % 50 == 0)

        env = AtariPreprocessing(env, frame_skip=config["FIXED_FRAMESKIP"], scale_obs=True, grayscale_obs=True)
        env = FrameStack(env, config["FRAME_STACK_SIZE"])

        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk

def make_video_env(config, seed):
    env = gym.make(config["ENV_NAME"], frameskip=1, render_mode="rgb_array", repeat_action_probability=0.0)
    env = AtariPreprocessing(env, frame_skip=config["FIXED_FRAMESKIP"], scale_obs=True, grayscale_obs=True)
    env = FrameStack(env, config["FRAME_STACK_SIZE"])
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def record_episode_and_save_gif(model, env, device, gif_path):
    frames = []
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        with torch.no_grad():
            obs_np = np.array(obs)
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _, _ = model.get_action_and_value(obs_tensor)
        
        frame = env.render()
        frames.append(frame)
        
        obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward
    frames.append(env.render())
    
    if frames:
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Saved GIF: {gif_path}, recorded {len(frames)} frames")
    
    return total_reward