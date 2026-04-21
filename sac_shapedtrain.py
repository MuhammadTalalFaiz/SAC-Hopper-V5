import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

# Create the environment
env = gym.make("Hopper-v5")

# Set up TensorBoard logging
log_dir = "./logs/sac_hopper/"
writer = SummaryWriter(log_dir)  # Create the SummaryWriter

class TQDMProgressBar(BaseCallback):
    """
    A custom callback to show a progress bar with tqdm.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(TQDMProgressBar, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = None
        

    def _on_training_start(self) -> None:
        self.progress_bar = tqdm(total=self.total_timesteps, desc="Training Progress", leave=True)

    def _on_step(self) -> bool:
        # Update the progress bar on each step
        self.progress_bar.update(self.model.n_envs)

        return True

        

    def _on_training_end(self) -> None:
        self.progress_bar.close()

def reward_shaping(obs, action, reward, forward_velocity):
    torso_angle = obs[1]  # Torso angle is in the second element of the observation
    
    # Define the penalties
    velocity_penalty = -0.1 * abs(forward_velocity - 2.0)  # Encourage velocity close to 2.0
    stability_penalty = -0.1 * abs(torso_angle)  # Encourage torso angle to be small

    # Shaped reward: original reward + penalties
    shaped_reward = reward + stability_penalty + velocity_penalty
    return shaped_reward

total_timesteps = 250000

# Wrapping the environment to include reward shaping
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, writer=None):
        super(RewardShapingWrapper, self).__init__(env)
        self.writer = writer  # Pass writer explicitly
        self.episode_rewards = []  # List to track the total rewards per episode
        self.episode_forward_velocities = []  # List to track forward velocities per episode
        self.elapsed_timesteps = 0  # Initialize overall elapsed timesteps

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        forward_velocity = obs[5]  # Assuming forward velocity is in the 6th element of the observation
        
        # Apply the reward shaping
        shaped_reward = reward_shaping(obs, action, reward, forward_velocity)
        
        # Increment the overall timestep counter
        self.elapsed_timesteps += 1
        
        # Log the shaped reward and forward velocity using SummaryWriter
        if self.writer is not None and self.elapsed_timesteps%100==0:
            # Use self.elapsed_timesteps for step-wise logging
            self.writer.add_scalar('rollout/forward_velocity', forward_velocity, self.elapsed_timesteps)
            self.writer.add_scalar('rollout/shaped_reward', shaped_reward, self.elapsed_timesteps)
        
        # Optionally track rewards and velocities for analysis later
        self.episode_rewards.append(shaped_reward)
        self.episode_forward_velocities.append(forward_velocity)
        
        if done or truncated:
            # At the end of an episode, log the statistics
            self.writer.add_scalar('episode/total_reward', sum(self.episode_rewards), self.elapsed_timesteps)
            self.writer.add_scalar('episode/mean_forward_velocity', np.mean(self.episode_forward_velocities), self.elapsed_timesteps)
            self.episode_rewards = []  # Reset for next episode
            self.episode_forward_velocities = []  # Reset for next episode
        
        return obs, shaped_reward, done, truncated, info

# Wrap the environment to apply reward shaping
env = RewardShapingWrapper(env, writer=writer)

# Create the SAC model with the extracted hyperparameters
model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-3,  # Learning rate
    gamma=0.99,  # Discount factor
    buffer_size=1000000,  # Replay buffer size
    tau=0.005,  # Target smoothing coefficient
    train_freq=(1, "step"),  # Training frequency
    gradient_steps=1,  # Gradient steps per update
    batch_size=64,  # Batch size for training
    learning_starts=10000,  # Timesteps before training starts
    tensorboard_log=log_dir,  # TensorBoard logging
    device='cuda'  # Use GPU for training
)

# Train the model
model.learn(total_timesteps=total_timesteps,callback=TQDMProgressBar(total_timesteps))

# Save the model
model.save("sac_hopper")
