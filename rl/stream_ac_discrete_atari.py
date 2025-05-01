import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Categorical
from optim import ObGD as Optimizer
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import torch.nn.functional as F
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init

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

class StreamAC(nn.Module):
    def __init__(self, n_actions=3, hidden_size=256, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.network_value = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=5),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=3),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(256, hidden_size),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.network_policy = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=5),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=3),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(256, hidden_size),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        self.apply(initialize_weights)
        self.optimizer_policy = Optimizer(self.network_policy.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_value = Optimizer(self.network_value.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def pi(self, x):
        x = torch.tensor(np.array(x), dtype=torch.float)
        preferences = self.network_policy(x)
        probs = F.softmax(preferences, dim=-1)
        return probs

    def v(self, x):
        x = torch.tensor(np.array(x), dtype=torch.float)
        return self.network_value(x)

    def sample_action(self, s):
        probs = self.pi(s)
        dist = Categorical(probs)
        return dist.sample().numpy()

    def update_params(self, s, a, r, s_prime, done, entropy_coeff, overshooting_info=False):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float64), torch.tensor(np.array(a)), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float64), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float64)

        v_s, v_prime = self.v(s), self.v(s_prime)
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        probs = self.pi(s)
        dist = Categorical(probs)

        log_prob_pi = -(dist.log_prob(a)).sum()
        value_output = -v_s
        entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta).item()
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        self.optimizer_policy.step(delta.item(), reset=done)
        self.optimizer_value.step(delta.item(), reset=done)

        if overshooting_info:
            v_s, v_prime = self.v(s), self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta_bar = td_target - v_s
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")

def main(env_name, seed, lr, gamma, lamda, total_steps, entropy_coeff, kappa_policy, kappa_value, debug, overshooting_info, render=False):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = NormalizeObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env = ScaleReward(env, gamma=gamma)
    agent = StreamAC(n_actions=env.action_space.n, lr=lr, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    episode_num = 1
    for t in range(1, total_steps+1):
        a = agent.sample_action(s)
        s_prime, r, terminated, _, info = env.step(a)
        agent.update_params(s, a, r, s_prime, terminated, entropy_coeff, overshooting_info)
        s = s_prime
        if info and "episode" in info:
            if debug:
                print("Episodic Return: {}, Time Step {}, Episode Number {}".format(info['episode']['r'][0], t, episode_num))
            returns.append(info['episode']['r'][0])
            term_time_steps.append(t)
            s, _ = env.reset()
            episode_num += 1
    env.close()
    save_dir = "data_stream_ac_{}_lr{}_gamma{}_lamda{}_entropy_coeff{}".format(env.spec.id, lr, gamma, lamda, entropy_coeff)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=10_000_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args.env_name, args.seed, args.lr, args.gamma, args.lamda, args.total_steps, args.entropy_coeff, args.kappa_policy, args.kappa_value, args.debug, args.overshooting_info, args.render)