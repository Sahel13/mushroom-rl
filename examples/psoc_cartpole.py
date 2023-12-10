import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import trange

from mushroom_rl.core import Core, Logger
from mushroom_rl import environments as envs
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.algorithms.actor_critic import TRPO, PPO

from mushroom_rl.policy import GaussianTorchPolicy
import os
import pandas as pd


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, **kwargs):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        return self._h3(features2)


def experiment(
    alg, alg_name, n_epochs, n_steps, n_steps_per_fit, n_eval_episodes, alg_params, policy_params, seed
):
    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: " + alg.__name__)

    mdp = envs.OurCartpole(seed=seed)

    critic_params = dict(
        network=Network,
        optimizer={"class": optim.Adam, "params": {"lr": 3e-4}},
        loss=F.mse_loss,
        n_features=256,
        batch_size=64,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(1,),
    )

    policy = GaussianTorchPolicy(
        Network,
        mdp.info.observation_space.shape,
        mdp.info.action_space.shape,
        **policy_params,
    )

    alg_params["critic_params"] = critic_params

    agent = alg(mdp.info, policy, **alg_params)

    core = Core(agent, mdp)

    reward_list = []

    dataset = core.evaluate(n_episodes=n_eval_episodes, render=False)
    R = np.mean(dataset.undiscounted_return)
    reward_list.append((0, R))

    logger.epoch_info(0, R=R)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_eval_episodes, render=False)
        R = np.mean(dataset.undiscounted_return)

        logger.epoch_info(it + 1, R=R)
        reward_list.append(((it + 1) * n_steps, R))

    csv_file = f"{alg_name}_cartpole.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        new_column = pd.DataFrame([reward[1] for reward in reward_list],
                                  columns=[f"Seed {seed}"])
        df = df.merge(new_column, left_index=True, right_index=True)
        df.to_csv(csv_file, index=False)
    else:
        df = pd.DataFrame(reward_list, columns=["n_samples", f"Seed {seed}"])
        df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    max_kl = 0.015

    policy_params = dict(std_0=2.5, n_features=256)

    ppo_params = dict(
        actor_optimizer={"class": optim.Adam, "params": {"lr": 3e-4}},
        n_epochs_policy=4,
        batch_size=64,
        eps_ppo=0.1,
        lam=0.95,
    )

    trpo_params = dict(
        ent_coeff=0.0,
        max_kl=0.01,
        lam=0.95,
        n_epochs_line_search=10,
        n_epochs_cg=100,
        cg_damping=1e-2,
        cg_residual_tol=1e-10,
    )

    algs_params = [(TRPO, "trpo", trpo_params), (PPO, "ppo", ppo_params)]

    for alg, alg_name, alg_params in algs_params:
        for seed in range(1, 11):
            experiment(
                alg=alg,
                alg_name=alg_name,
                n_epochs=20,
                n_steps=15_000,
                n_steps_per_fit=3000,
                n_eval_episodes=30,
                alg_params=alg_params,
                policy_params=policy_params,
                seed=seed
            )
