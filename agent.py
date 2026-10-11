import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory 
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random

from collections import deque

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else: 
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")
        self.CHECKPOINT_FILE = os.path.join(RUNS_DIR, f"{self.param_set}_checkpoint.pt")


    def run(self, is_training=True, render=False, resume=False):
        # use_lidar=False uses the clean 12-dimensional kinematic state vector
        env = gym.make("FlappyBird-v0", use_lidar=False, render_mode="human" if render else None)

        num_states = env.observation_space.shape[0] # input dim (12)
        num_actions = env.action_space.n # output dim (2)

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            start_episode = 0
            global_steps = 0
            best_reward = float("-inf")
            recent_rewards = deque(maxlen=50)

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            # If resume flag is set, load checkpoint state or previous weights
            if resume:
                if os.path.exists(self.CHECKPOINT_FILE):
                    checkpoint = torch.load(self.CHECKPOINT_FILE, map_location=device, weights_only=False)
                    policy_dqn.load_state_dict(checkpoint["model_state_dict"])
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    start_episode = checkpoint.get("episode", 0) + 1
                    global_steps = checkpoint.get("global_steps", 0)
                    epsilon = checkpoint.get("epsilon", self.epsilon_min)
                    best_reward = checkpoint.get("best_reward", float("-inf"))
                    print(f"Resuming training from Episode {start_episode+1} | Epsilon: {epsilon:.4f} | Best Reward: {best_reward:.1f}")
                elif os.path.exists(self.MODEL_FILE):
                    policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device, weights_only=True))
                    epsilon = max(self.epsilon_min, 0.1) # low exploration for fine-tuning
                    print(f"Loaded weights from {self.MODEL_FILE}. Continuing training with epsilon={epsilon:.4f}...")
                else:
                    print(f"No previous checkpoint found. Starting fresh training...")

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            target_dqn.eval()

        else:
            if not os.path.exists(self.MODEL_FILE):
                raise FileNotFoundError(f"Model file '{self.MODEL_FILE}' not found. Please train the model first.")
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device, weights_only=True))
            policy_dqn.eval()
            start_episode = 0

        for episode in itertools.count(start=start_episode):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_reward = 0
            terminated = False

            while not terminated and episode_reward < self.reward_threshold:
                if is_training and random.random() < epsilon:
                    action_scalar = env.action_space.sample() # explore
                    action = torch.tensor(action_scalar, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).argmax(dim=1).squeeze()
                    action_scalar = action.item()

                next_state, reward, terminated, _, _ = env.step(action_scalar)
                episode_reward += reward

                # Create tensors
                reward_tensor = torch.tensor(reward, dtype=torch.float, device=device)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float, device=device)

                if is_training:
                    action_tensor = torch.tensor(action_scalar, dtype=torch.long, device=device)
                    memory.append((state, action_tensor, next_state_tensor, reward_tensor, terminated))
                    global_steps += 1

                    # Optimize per step once replay buffer has sufficient samples
                    if len(memory) >= self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                        # Sync target network every network_sync_rate steps
                        if global_steps % self.network_sync_rate == 0:
                            target_dqn.load_state_dict(policy_dqn.state_dict())

                state = next_state_tensor

            if is_training:
                recent_rewards.append(episode_reward)
                avg_reward = sum(recent_rewards) / len(recent_rewards)

                # Decay epsilon per episode
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                print(f"Episode {episode+1:5d} | Reward: {episode_reward:6.1f} | Avg(50): {avg_reward:6.1f} | Epsilon: {epsilon:.4f} | Steps: {global_steps}")

                # Save best model
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    log_msg = f"New best reward = {episode_reward:.2f} for episode={episode+1} (steps={global_steps}, avg={avg_reward:.2f})"
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    print(f"  >>> Saved new best model to {self.MODEL_FILE}")

                # Save checkpoint state periodically (every 50 episodes) and on new best
                if (episode + 1) % 50 == 0 or episode_reward >= best_reward:
                    checkpoint_data = {
                        "model_state_dict": policy_dqn.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "episode": episode,
                        "global_steps": global_steps,
                        "epsilon": epsilon,
                        "best_reward": best_reward,
                    }
                    torch.save(checkpoint_data, self.CHECKPOINT_FILE)
            else:
                print(f"Test Episode {episode+1} | Reward: {episode_reward:.1f}")


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Get batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        # Calculate target Q-values
        with torch.no_grad():
            next_q = target_dqn(next_states).max(dim=1)[0]
            target_q = rewards + (1.0 - terminations) * self.gamma * next_q

        # Calculate predicted Q-value from policy network
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(-1)

        # Compute Smooth L1 (Huber) loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize model with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
        self.optimizer.step()


if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--resume', help='Resume training from existing checkpoint', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True, resume=args.resume)
    else:
        dql.run(is_training=False, render=True)