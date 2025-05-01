import argparse, os
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from optim import ObGD as Optimizer
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init
import matplotlib.pyplot as plt
import pandas as pd

class Trace(gym.Wrapper):
    def __init__(self, shape=(), beta=0.99):
        self.mean = np.zeros(shape, "float64")
        self.beta = beta
        self.count = 0

    def update(self, x):
        self.count += 1
        self.mean = self.beta * self.mean + (1 - self.beta) * x
        return self.mean / (1 - self.beta ** self.count)

    def reset(self):
        self.mean = np.zeros_like(self.mean)
        self.count = 0

class ObservationTraces(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, beta: float = 0.999):
        gym.utils.RecordConstructorArgs.__init__(self, beta=beta)
        gym.Wrapper.__init__(self, env)
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            self.trace = Trace(shape=self.single_observation_space.shape, beta=beta)
        else:
            self.trace = Trace(shape=self.observation_space.shape, beta=beta)

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.get_trace(obs)
        else:
            obs = self.get_trace(np.array([obs]))[0]
        if terminateds or truncateds:
            self.trace.reset()
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.is_vector_env:
            obs = self.get_trace(obs)
            return obs, info
        else:
            obs = self.get_trace(np.array([obs]))[0]
            return obs, info

    def get_trace(self, obs):
        return self.trace.update(obs)

class ETTEnvironment(gym.Env):
    def __init__(self, dataset_path="ETTm2.csv"):
        super(ETTEnvironment, self).__init__()
        # download the dataset from https://github.com/zhouhaoyi/ETDataset
        download_url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm2.csv"
        if not os.path.exists(dataset_path):
            import urllib.request
            urllib.request.urlretrieve(download_url, dataset_path)
        self.current_step = 0
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.spec = gym.envs.registration.EnvSpec("ETTEnvironment-v0")
        self.df = self.process_data(dataset_path)

    def process_data(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        self.df = self.df.drop(columns=["date"])
        self.df = self.df.astype(np.float64)
        self.df["original_cumulant"] = self.df.iloc[:, -1]
        self.scaling_value = self.df.iloc[:, -1].max() - self.df.iloc[:, -1].min()
        self.add_value = self.df.iloc[:, -1].min()
        self.df.iloc[:, -1] = (self.df.iloc[:, -1] - self.add_value) / self.scaling_value
        return self.df

    def reset(self, seed=None, options={}):
        self.current_step = 0
        self.max_steps = len(self.df) - 1.0
        return self._get_observation(), {}

    def step(self, action=None):
        self.current_step += 1
        observation = self._get_observation()
        reward = self._calculate_reward()
        done = self.current_step >= self.max_steps
        return observation, reward, done, 0, {}

    def _get_observation(self):
        return self.df.iloc[self.current_step, :-1].values

    def _calculate_reward(self):
        return self.df.iloc[self.current_step+1, -1]

    def close(self):
        os.remove("ETTm2.csv")

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class StreamTD(nn.Module):
    def __init__(self, n_obs=7, hidden_size=128, lr=1.0, gamma=0.9, lamda=0.9, kappa_value=2.0):
        super(StreamTD, self).__init__()
        self.gamma = gamma
        self.fc1_v   = nn.Linear(n_obs, hidden_size)
        self.hidden_v  = nn.Linear(hidden_size, hidden_size)
        self.fc_v  = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)
        self.optimizer = Optimizer(list(self.parameters()), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def v(self, x):
        x = self.fc1_v(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_v(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        return self.fc_v(x)

    def predict(self, s):
        s = torch.tensor(np.array(s), dtype=torch.float)
        return self.v(s).item()

    def update_params(self, s, r, s_prime, done, overshooting_info=False):
        done_mask = 0 if done else 1
        s, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), torch.tensor(np.array(r)),\
                                   torch.tensor(np.array(s_prime), dtype=torch.float), torch.tensor(np.array(done_mask), dtype=torch.float)

        v_s = self.v(s)
        td_target = r + self.gamma * self.v(s_prime) * done_mask
        delta = td_target - v_s
        value_output = -v_s
        self.optimizer.zero_grad()
        value_output.backward()
        self.optimizer.step(delta.item(), reset=done)

        if overshooting_info:
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta_bar = td_target - self.v(s)
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")

def main(seed, lr, gamma, lamda, total_steps, kappa_value, debug, overshooting_info):
    torch.manual_seed(seed); np.random.seed(seed)
    env = ETTEnvironment()
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = ObservationTraces(env, beta=0.999)
    env = NormalizeObservation(env)
    env = ScaleReward(env, gamma=gamma)
    agent = StreamTD(n_obs=env.observation_space.shape[0], lr=lr, gamma=gamma, lamda=lamda, kappa_value=kappa_value)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))
    s, _ = env.reset()
    cumulants, predictions = [], []
    for _ in range(total_steps):
        s_prime, c, terminated, _, _ = env.step(None)
        agent.update_params(s, c, s_prime, terminated, overshooting_info=overshooting_info)
        s = s_prime
        predictions.append(agent.predict(s) * np.sqrt(env.reward_stats.var + 1e-8).squeeze())
        cumulants.append(c * np.sqrt(env.reward_stats.var + 1e-8).squeeze())
        if terminated:
            s, _ = env.reset()
            break
    env.close()

    # compute actual returns using formula: G_t = r_t + gamma * G_{t+1} starting with G_T = 0
    reversed_actual_returns = []
    return_t = 0
    for t in reversed(range(total_steps)):
        return_t = return_t * gamma + cumulants[t]
        reversed_actual_returns.append(return_t)
    actual_returns = list(reversed(reversed_actual_returns))

    plt.figure(figsize=(12, 4))
    plt.plot(actual_returns, label="Actual Return", linewidth=3.0, color="tab:green")
    plt.plot(predictions, label="Prediction", linewidth=3.0, color="tab:blue")
    plt.xlim([0, 5000])
    plt.xlabel("Time Step", fontsize=20)
    plt.ylabel("Normalized Oil Temp.", fontsize=20)
    plt.legend()
    plt.savefig("td_ettm2_start.pdf", bbox_inches='tight')

    plt.figure(figsize=(12, 4))
    plt.plot(actual_returns, label="Actual Return", linewidth=3.0, color="tab:green")
    plt.plot(predictions, label="Prediction", linewidth=3.0, color="tab:blue")
    plt.xlim([total_steps-5000, total_steps])
    plt.ylim([35, 85])
    plt.xlabel("Time Step", fontsize=20)
    plt.ylabel("Normalized Oil Temp.", fontsize=20)
    plt.legend()
    plt.savefig("td_ettm2_end.pdf", bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream TD(λ)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--total_steps', type=int, default=68_000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    args = parser.parse_args()
    main(args.seed, args.lr, args.gamma, args.lamda, args.total_steps, args.kappa_value, args.debug, args.overshooting_info)