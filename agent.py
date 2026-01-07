import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import flappy_bird_gymnasium
except Exception:
    flappy_bird_gymnasium = None

import gymnasium
import torch
from torch import nn
import random
import itertools
import yaml
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from replay_memory import ReplayMemory
from dqn import DQN
from datetime import datetime, timedelta

# -------------------- CONFIG --------------------

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use("Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- AGENT --------------------

class Agent:
    def __init__(self, hyperparameter_set):
        self.hyperparameter_set = hyperparameter_set

        with open("hyperparameters.yml", "r") as file:
            hyperparameters = yaml.safe_load(file)[hyperparameter_set]

        self.env_id = hyperparameters["env_id"]
        self.env_make_params = hyperparameters.get("env_make_params", {})
        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.discount_factor_g = hyperparameters["discount_factor_g"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.stop_on_reward = hyperparameters["stop_on_reward"]
        self.fc1_nodes = hyperparameters["fc1_nodes"]
        self.enable_double_dqn = hyperparameters["enable_double_dqn"]
        self.enable_dueling_dqn = hyperparameters["enable_dueling_dqn"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.pt")
        self.FINAL_MODEL_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.png")

    # -------------------- RUN --------------------

    def run(self, is_training=True, render=False):
        env = gymnasium.make(
            self.env_id,
            render_mode="human" if render else None,
            **self.env_make_params
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(
            num_states,
            num_actions,
            hidden_dim=self.fc1_nodes,
            enable_dueling_dqn=self.enable_dueling_dqn
        ).to(device)

        if is_training:
            target_dqn = DQN(
                num_states,
                num_actions,
                hidden_dim=self.fc1_nodes,
                enable_dueling_dqn=self.enable_dueling_dqn
            ).to(device)

            target_dqn.load_state_dict(policy_dqn.state_dict())
            target_dqn.eval()

            memory = ReplayMemory(self.replay_memory_size, seed=42)
            self.optimizer = torch.optim.Adam(
                policy_dqn.parameters(), lr=self.learning_rate
            )

            epsilon = self.epsilon_init
            epsilon_history = []
            env_step_count = 0
            target_sync_count = 0
            last_graph_update_time = datetime.now()
            reward_list_mean = []

        else:
            policy_dqn.load_state_dict(torch.load(self.FINAL_MODEL_FILE))
            policy_dqn.eval()

        rewards_per_episode = []

        # -------------------- EPISODES --------------------

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            episode_reward = 0.0
            done = False

            while not done:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                reward = float(np.clip(reward, -100, 100))
                episode_reward += reward

                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

                if is_training:
                    memory.append((
                        state,
                        torch.tensor(action, device=device),
                        torch.tensor(reward, device=device),
                        next_state,
                        done
                    ))

                state = next_state
                if is_training:
                    env_step_count += 1

                if is_training and env_step_count % 4 == 0 and len(memory) >= self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(policy_dqn, target_dqn, mini_batch)
                    target_sync_count += 1

                    if target_sync_count >= self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        target_sync_count = 0

            rewards_per_episode.append(episode_reward)

            if is_training:
                print(
                    f"Episode {episode} | "
                    f"Reward {episode_reward:.1f} | "
                    f"Epsilon {epsilon:.3f}"
                )
            else:
                print(
                    f"Episode {episode} | "
                    f"Reward {episode_reward:.1f}"
                )


            if is_training:
                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
                epsilon_history.append(epsilon)

                if (
                    len(rewards_per_episode) >= 10
                    and np.mean(rewards_per_episode[-10:]) >= 200
                    and (
                        np.mean(rewards_per_episode[-10:])
                        >= (reward_list_mean[-1] if reward_list_mean else -float("inf"))
                    )
                ):
                    reward_list_mean.append(np.mean(rewards_per_episode[-10:]))

                    log_message = (
                        f"{datetime.now().strftime(DATE_FORMAT)} | "
                        f"Episode {episode} | "
                        f"Mean(10) {np.mean(rewards_per_episode[-10:]):.1f} | "
                        f"Saved model"
                    )
                    print(log_message)

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_message + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                if datetime.now() - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = datetime.now()

        if is_training:
            torch.save(policy_dqn.state_dict(), self.FINAL_MODEL_FILE)

    # -------------------- OPTIMIZE --------------------

    def optimize(self, policy_dqn, target_dqn, mini_batch):
        states, actions, rewards, new_states, dones = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        new_states = torch.stack(new_states)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (~dones).float() * self.discount_factor_g * \
                    target_dqn(new_states).gather(1, best_actions.unsqueeze(1)).squeeze()
            else:
                target_q = rewards + (~dones).float() * self.discount_factor_g * \
                    target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # -------------------- GRAPH --------------------

    def save_graph(self, rewards, epsilon_history):
        fig = plt.figure(figsize=(12, 5))

        mean_rewards = [
            np.mean(rewards[max(0, i - 99):i + 1])
            for i in range(len(rewards))
        ]

        plt.subplot(121)
        plt.plot(mean_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Mean Reward (100 ep)")

        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.xlabel("Episodes")
        plt.ylabel("Epsilon")

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

# -------------------- ENTRY --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hyperparameters")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    agent = Agent(args.hyperparameters)

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)
