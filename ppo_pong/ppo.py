import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from collections import deque
import wandb
import os
import imageio
from model import ActorCritic
from utils import make_env, make_video_env, record_episode_and_save_gif

config = {
    "ENV_NAME": "ALE/Pong-v5",
    "NUM_ENVS": 8,                # Number of parallel environments to run
    "TOTAL_TIMESTEPS": 3_000_000, # Number of agent-environment interactions
    "HORIZON_T": 128,             # Number of steps to collect from each environemnt per rollout
    "NUM_EPOCHS": 4,              # Number of times to iterate over the collected rollout data
    "MINIBATCH_SIZE": 256,        # Size of minibatches for the PPO update
    "GAMMA": 0.99,                # Discount factor for future rewards
    "GAE_LAMBDA": 0.95,           # Lambda for GAE
    "CLIP_EPS_START": 0.1,        # Intitial clipping for preventing aggresive updates
    "VF_COEF": 0.5,               # Coefficient for the value function loss
    "ENTROPY_COEF": 0.01,         # Coefficient for the entropy loss (encourages exploration)
    "LR_START": 2.5e-4,           # Initial learning rate
    "FRAME_STACK_SIZE": 4,        # Number of consecutive frames to stack
    "IMAGE_SIZE": (84, 84),       # Size of the preprocessed game frames 
    "SEED": 42,                   # Random seed for reproducibility
    "NORMALIZE_ADVANTAGE": True,
    "FIXED_FRAMESKIP": 4,         # Number of frames to skip per action
    "VIDEO_CHECK_FREQ": 50,       # How often in updates to check for recording best GIF
    "BEST_SCORE_THRESHOLD": 11.0  # Minimum score for GIF
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment tracking and visualization
wandb.init(
    project="ppo-pong-atari",
    config=config,
    name=f"ppo_{config['ENV_NAME']}_{np.random.randint(1000)}",
    save_code=True,
)


# Computes Generalized Advantage Estimation (GAE) and the value targets (returns)
def compute_gae(next_value, rewards, dones, values, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards).to(DEVICE)
    last_gae_lam = 0
    # Iterates backwards through the rollout data
    for t in reversed(range(config["HORIZON_T"])):
        if t == config["HORIZON_T"] - 1:
            next_non_terminal = 1.0 - dones_buffer[t]
            next_values_for_gae = next_value
        else:
            next_non_terminal = 1.0 - dones_buffer[t]
            next_values_for_gae = values[t+1]
        # Calculate the TD error
        delta = rewards[t] + gamma * next_values_for_gae * next_non_terminal - values[t]

        # Recursively calculate the GAE for the current timestep
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

    # The returns (value targets) are the advantages plus the old value estimates
    returns = advantages + values
    return advantages, returns

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Run name: {wandb.run.name}")

    gif_dir = "training_gifs"
    os.makedirs(gif_dir, exist_ok=True)
    print(f"GIFs in folder: {gif_dir}")

    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(config["SEED"])

    # Create a vectorized environment that runs NUM_ENVS parallel games.
    envs = SyncVectorEnv(
        [make_env(config, seed=config["SEED"], idx=i, capture_video=False, run_name=wandb.run.name) for i in range(config["NUM_ENVS"])]
    )
    num_actions = envs.single_action_space.n

    # Model and optimizer initialization
    model = ActorCritic(num_actions, config).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config["LR_START"], eps=1e-5)

    # Buffers storing the agent's experiences for one full rollout
    obs_shape = envs.single_observation_space.shape
    obs_buffer = torch.zeros((config["HORIZON_T"], config["NUM_ENVS"]) + obs_shape).to(DEVICE)
    actions_buffer = torch.zeros((config["HORIZON_T"], config["NUM_ENVS"]) + envs.single_action_space.shape).to(DEVICE)
    logprobs_buffer = torch.zeros((config["HORIZON_T"], config["NUM_ENVS"])).to(DEVICE)
    rewards_buffer = torch.zeros((config["HORIZON_T"], config["NUM_ENVS"])).to(DEVICE)
    dones_buffer = torch.zeros((config["HORIZON_T"], config["NUM_ENVS"])).to(DEVICE)
    values_buffer = torch.zeros((config["HORIZON_T"], config["NUM_ENVS"])).to(DEVICE)

    # Training loop state initialization
    global_step = 0
    num_updates = config["TOTAL_TIMESTEPS"] // (config["HORIZON_T"] * config["NUM_ENVS"])
    next_obs_tuple, info = envs.reset(seed=config["SEED"])
    next_obs = torch.Tensor(np.array(next_obs_tuple)).to(DEVICE)
    next_done = torch.zeros(config["NUM_ENVS"]).to(DEVICE)
    recent_scores = deque(maxlen=100)
    best_score = -float('inf')
    video_env = None
    gif_counter = 0

    # Main training loop
    for update in range(1, num_updates + 1):
        # Decrease learning rate and clipping epsilon
        frac = 1.0 - (update - 1.0) / num_updates
        current_lr = config["LR_START"] * frac
        current_clip_eps = config["CLIP_EPS_START"] * frac
        optimizer.param_groups[0]["lr"] = current_lr

        # Agent interacts with parallel environments for HORIZON_T steps
        for step in range(config["HORIZON_T"]):
            global_step += config["NUM_ENVS"]

            # Store the current observation and done state
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done

            # Get action and value from the model
            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_obs)
                values_buffer[step] = value.flatten()

            # Store the action and its log probability
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob

            # Execute the action in the environments
            next_obs_tuple, reward, terminated, truncated, info_dict = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            # Store the received reward and update the next state and done flag
            rewards_buffer[step] = torch.tensor(reward).to(DEVICE).view(-1)
            next_obs_numpy = np.array(next_obs_tuple)
            next_obs = torch.Tensor(next_obs_numpy).to(DEVICE)
            next_done = torch.Tensor(done).to(DEVICE)

            # Check if any episode ended and log their stats
            if "final_info" in info_dict:
                for i, item in enumerate(info_dict["_final_info"]):
                    if item:
                        ep_info = info_dict["final_info"][i]
                        if ep_info and "episode" in ep_info:
                             episodic_return = ep_info['episode']['r']
                             episodic_length = ep_info['episode']['l']
                             
                             if isinstance(episodic_return, np.ndarray):
                                 episodic_return = episodic_return.item()
                             if isinstance(episodic_length, np.ndarray):
                                 episodic_length = episodic_length.item()
                             
                             recent_scores.append(episodic_return)
                             avg_score_100 = np.mean(recent_scores) if recent_scores else -21.0
                             
                             if isinstance(avg_score_100, np.ndarray):
                                 avg_score_100 = avg_score_100.item()
                             
                             print(f"global_step={global_step}, env_id={i}, episodic_return={episodic_return:.2f}, len={episodic_length}, avg_score_100={avg_score_100:.2f}")

        # Advantage and return calculation
        with torch.no_grad():
            # "Bootstrap" the value estimate for the state after rollout
            _, next_value_bootstrap = model(next_obs)
            next_value_bootstrap = next_value_bootstrap.reshape(1, -1)
            advantages, returns = compute_gae(next_value_bootstrap, rewards_buffer, dones_buffer, values_buffer, config["GAMMA"], config["GAE_LAMBDA"])

        # Flatten the rollout data
        b_obs = obs_buffer.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Iterate over the collected data
        batch_size_total = config["HORIZON_T"] * config["NUM_ENVS"]
        inds = np.arange(batch_size_total)
        for epoch in range(config["NUM_EPOCHS"]):
            np.random.shuffle(inds)
            for start in range(0, batch_size_total, config["MINIBATCH_SIZE"]):
                end = start + config["MINIBATCH_SIZE"]
                minibatch_inds = inds[start:end]

                # Get data for current minibatch
                mb_obs = b_obs[minibatch_inds]
                mb_actions = b_actions[minibatch_inds]
                mb_logprobs_old = b_logprobs[minibatch_inds]
                mb_advantages = b_advantages[minibatch_inds]
                mb_returns = b_returns[minibatch_inds]

                # Reevaluate actions and values for current policy
                _, new_logprobs, entropy, new_values = model.get_action_and_value(mb_obs, mb_actions.long())
                new_values = new_values.view(-1)

                logratio = new_logprobs - mb_logprobs_old
                ratio = torch.exp(logratio)

                # Normalize advantages within the minibatch for stable updates
                mb_advantages_norm = mb_advantages
                if config["NORMALIZE_ADVANTAGE"]:
                    mb_advantages_norm = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Calculate the PPO clipped surrogate objective loss
                pg_loss1 = -mb_advantages_norm * ratio
                pg_loss2 = -mb_advantages_norm * torch.clamp(ratio, 1 - current_clip_eps, 1 + current_clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Calculate the value function (critic) and the entropy loss
                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                
                # Combine losses
                loss = pg_loss - config["ENTROPY_COEF"] * entropy_loss + config["VF_COEF"] * v_loss

                # Perform the optimization step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # Logging and diagnostic
        avg_score_100_update = np.mean(recent_scores) if recent_scores else -21.0
        if isinstance(avg_score_100_update, np.ndarray):
            avg_score_100_update = avg_score_100_update.item()
            
        # Calculate how well the value function predicts the returns
        y_pred, y_true = values_buffer.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log metrics to Weights & Biases
        wandb.log({
            "charts/avg_score_100": avg_score_100_update,
            "charts/learning_rate": current_lr,
            "charts/clip_epsilon": current_clip_eps,
            "losses/total_loss": loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/entropy_loss": entropy_loss.item(),
            "metrics/explained_variance": explained_var,
        }, step=global_step)

        # Print progress
        if (update % 20 == 0) or (update == num_updates):
            print(f"Update {update}/{num_updates}, Global Timesteps: {global_step}, Avg Score (last 100): {avg_score_100_update:.2f}")
            print(f"LR: {current_lr:.2e}, Clip Eps: {current_clip_eps:.3f}")
            print(f"Policy Loss: {pg_loss.item():.4f}, Value Loss: {v_loss.item():.4f}, Entropy: {entropy_loss.item():.4f}")

        # Model and video saving
        if (update % config["VIDEO_CHECK_FREQ"] == 0 and 
            avg_score_100_update > best_score and 
            avg_score_100_update > config["BEST_SCORE_THRESHOLD"]):
            
            best_score = avg_score_100_update
            gif_counter += 1
            
            print(f"New best score: {best_score:.2f}, recording GIF...")
            
            if video_env is None:
                video_env = make_video_env(config, config["SEED"])
            
            gif_path = os.path.join(gif_dir, f"best_episode_update_{update}_score_{best_score:.1f}.gif")
            
            episode_reward = record_episode_and_save_gif(model, video_env, DEVICE, gif_path)
            
            print(f"Saved the best episode as GIF: {gif_path}, reward: {episode_reward:.2f}")
            
            model_path = os.path.join(gif_dir, f"best_model_update_{update}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved the best model: {model_path}")

    envs.close()
    if video_env is not None:
        video_env.close()
    
    final_model_path = "ppo_pong_final.pth"
    torch.save(model.state_dict(), final_model_path)
    wandb.save(final_model_path)
    
    print("Training ended.")
    wandb.finish()