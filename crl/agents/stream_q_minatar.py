import argparse
import os
import pickle

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from envs.minatar.continual_env import CL_envs_cycle
from optim import ObGD as Optimizer
from sparse_init import sparse_init
from wrappers.normalization_wrappers import (
    NormalizeObservation,
    ScaleReward,
)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class LayerNormalization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.layer_norm(input, input.size())

    def extra_repr(self) -> str:
        return "Layer Normalization"


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)


class StreamQ(nn.Module):
    def __init__(
        self,
        n_channels=4,
        n_actions=3,
        hidden_size=128,
        lr=1.0,
        epsilon_target=0.01,
        epsilon_start=1.0,
        exploration_fraction=0.1,
        total_steps=50_000_000,
        gamma=0.99,
        lamda=0.8,
        kappa_value=2.0,
    ):
        super(StreamQ, self).__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_target = epsilon_target
        self.epsilon = epsilon_start
        self.exploration_fraction = exploration_fraction
        self.total_steps = total_steps
        self.time_step = 0
        self.network = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, stride=1),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(1024, hidden_size),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.apply(initialize_weights)
        self.optimizer = Optimizer(
            list(self.parameters()),
            lr=lr,
            gamma=gamma,
            lamda=lamda,
            kappa=kappa_value,
        )

    def q(self, x):
        x = torch.moveaxis(x, -1, 0)
        return self.network(x)

    def sample_action(self, s):
        self.time_step += 1
        self.epsilon = linear_schedule(
            self.epsilon_start,
            self.epsilon_target,
            self.exploration_fraction * self.total_steps,
            self.time_step,
        )
        if isinstance(s, np.ndarray):
            s = torch.tensor(np.array(s), dtype=torch.float)
        if np.random.rand() < self.epsilon:
            q_values = self.q(s)
            greedy_action = torch.argmax(q_values, dim=-1).item()
            random_action = np.random.randint(0, self.n_actions)
            if greedy_action == random_action:
                return random_action, False
            else:
                return random_action, True
        else:
            q_values = self.q(s)
            return torch.argmax(q_values, dim=-1), False

    def update_params(
        self,
        s,
        a,
        r,
        s_prime,
        done,
        is_nongreedy,
        overshooting_info=False,
    ):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = (
            torch.tensor(np.array(s), dtype=torch.float),
            torch.tensor([a], dtype=torch.int).squeeze(0),
            torch.tensor(np.array(r)),
            torch.tensor(np.array(s_prime), dtype=torch.float),
            torch.tensor(np.array(done_mask), dtype=torch.float),
        )

        q_sa = self.q(s)[a]
        max_q_s_prime_a_prime = torch.max(self.q(s_prime), dim=-1).values
        td_target = r + self.gamma * max_q_s_prime_a_prime * done_mask
        delta = td_target - q_sa

        q_output = -q_sa
        self.optimizer.zero_grad()
        q_output.backward()
        self.optimizer.step(delta.item(), reset=(done or is_nongreedy))

        if overshooting_info:
            max_q_s_prime_a_prime = torch.max(self.q(s_prime), dim=-1).values
            td_target = r + self.gamma * max_q_s_prime_a_prime * done_mask
            delta_bar = td_target - self.q(s)[a]
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")


def main(
    env_name,
    seed,
    lr,
    gamma,
    lamda,
    total_steps,
    epsilon_target,
    epsilon_start,
    exploration_fraction,
    kappa_value,
    debug,
    overshooting_info,
    render=False,
):
    log_wandb = True
    if log_wandb:
        wandb.init(
            project="cssf",
            name="cycle",
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    env_idx = 0
    env = CL_envs_cycle(env_idx)

    def wrap_env(env):
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NormalizeObservation(env)
        env = ScaleReward(env, gamma=gamma)
        return env

    env = wrap_env(env)
    agent = StreamQ(
        n_channels=env.observation_space.shape[-1],
        n_actions=env.action_space.n,
        lr=lr,
        gamma=gamma,
        lamda=lamda,
        epsilon_target=epsilon_target,
        epsilon_start=epsilon_start,
        exploration_fraction=exploration_fraction,
        total_steps=total_steps,
        kappa_value=kappa_value,
    )
    # if debug:
    #     print("seed: {}".format(seed), "env: {}".format(env.spec.id))
    returns, term_time_steps = [], []
    s, _ = env.reset()
    episode_num = 1
    for t in range(1, total_steps + 1):
        if (t + 1) % (10**6) == 0:
            env_idx = env_idx + 1
            env = CL_envs_cycle(env_idx)
            env = wrap_env(env)
            s, _ = env.reset()
        a, is_nongreedy = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        agent.update_params(
            s,
            a,
            r,
            s_prime,
            terminated or truncated,
            is_nongreedy,
            overshooting_info,
        )
        s = s_prime
        if terminated or truncated:
            if debug:
                print(
                    "Episodic Return: {}, Time Step {}, Episode Number {}, Epsilon {}".format(
                        info["episode"]["r"][0],
                        t,
                        episode_num,
                        agent.epsilon,
                    )
                )
            returns.append(info["episode"]["r"][0])
            term_time_steps.append(t)
            terminated, truncated = False, False
            s, _ = env.reset()
            episode_num += 1
            avg_reward = np.average(returns[-100:])
            if log_wandb:
                wandb.log({"Reward": avg_reward}, step=t)
    env.close()
    save_dir = "data_stream_q_{}_lr{}_gamma{}_lamda{}".format(env.spec.id, lr, gamma, lamda)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(
        os.path.join(save_dir, "seed_{}.pkl".format(seed)),
        "wb",
    ) as f:
        pickle.dump((returns, term_time_steps, env_name), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Q(λ)")
    parser.add_argument("--env_name", type=str, default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.8)
    parser.add_argument("--epsilon_target", type=float, default=0.01)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--exploration_fraction", type=float, default=0.2)
    parser.add_argument("--kappa_value", type=float, default=2.0)
    parser.add_argument("--total_steps", type=int, default=50_000_000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overshooting_info", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    main(
        args.env_name,
        args.seed,
        args.lr,
        args.gamma,
        args.lamda,
        args.total_steps,
        args.epsilon_target,
        args.epsilon_start,
        args.exploration_fraction,
        args.kappa_value,
        args.debug,
        args.overshooting_info,
        args.render,
    )
