from typing import Any
from typing import Dict
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback
import minihack
from math import log10, floor
from sb3_ppolstm_minihack import MiniHackExtractor

import torch
from torch import nn
import gym

import optuna

N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = 200
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

ENV_ID = "MiniHack-River-v0"

DEFAULT_HYPERPARAMS = dict(policy="MultiInputLstmPolicy",
                           policy_kwargs=dict(
                               features_extractor_class=MiniHackExtractor, ))


def sample_ppo_lstm_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO-LSTM hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    n_steps = int(2**trial.suggest_int("n_steps", 3, 10))
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    ortho_init = bool(trial.suggest_categorical("ortho_init", [False, True]))
    net_arch = trial.suggest_categorical(
        "net_arch", ["tiny", "small", "shared", "shared_large"])
    activation_fn = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu", 'leaky_relu', 'elu'])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    enable_critic_lstm = trial.suggest_categorical('enable_critic_lstm',
                                                   [True, False])
    clip_range = 0.1 * trial.suggest_int("clip_range", 1, 4)
    lstm_hidden_size = 2**trial.suggest_int("lstm_hidden_size", 3, 10)
    vf_coef = float(trial.suggest_float("vf_coef", 1e-4, 1, log=True))
    batch_size = 2**trial.suggest_int("batch_size", 3, 9)


    net_arch = {
        "tiny": [{
            "pi": [64],
            "vf": [64]
        }],
        "small": [{
            "pi": [64, 64],
            "vf": [64, 64]
        }],
        "shared": [256],
        "shared_large": [512]
    }[net_arch]

    # Display true values
    trial.set_user_attr("gamma", gamma)
    trial.set_user_attr("gae_lambda", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)
    trial.set_user_attr("lstm_hidden_size", lstm_hidden_size)
    trial.set_user_attr("clip_range", clip_range)
    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("net_arch", net_arch)




    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU
    }[activation_fn]

    return {
        "batch_size": batch_size,
        "clip_range": clip_range,
        "vf_coef": vf_coef,
        "n_epochs": n_epochs,
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "enable_critic_lstm": enable_critic_lstm,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
            "lstm_hidden_size": lstm_hidden_size
        },
    }


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        prune: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.prune = prune
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.prune and self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial):

    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters
    kwargs.update(sample_ppo_lstm_params(trial))
    args = {
        'observation_keys': ['chars', 'chars_crop'],
        'penalty_time': -0.005,
        'penalty_step': -0.1
    }
    env = gym.make(ENV_ID, **args)
    env._max_episode_steps = 30

    model = RecurrentPPO(env=env, **kwargs)

    eval_env = gym.make(ENV_ID, **args)
    eval_env._max_episode_steps = 30

    eval_callback = TrialEvalCallback(eval_env,
                                      trial,
                                      n_eval_episodes=N_EVAL_EPISODES,
                                      eval_freq=EVAL_FREQ,
                                      deterministic=True)

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == '__main__':
    # Set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)

    sampler = optuna.samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP_TRIALS,
                                         n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(
        study_name='lunar_optuna',
        sampler=sampler,
        # pruner=pruner,
        storage='sqlite:///params.db',
        direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1, timeout=600)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        if key in trial.user_attrs:
            continue
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")
