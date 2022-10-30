import gym
import minihack
import time

import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO
from torch.optim import RMSprop

import torch
from torch import nn


def evaluate_lstm(model, eval_env):

    obs = eval_env.reset()

    # cell and hidden state of the LSTM
    lstm_states = None
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((1, ), dtype=bool)
    for _ in range(2000):
        action, lstm_states = model.predict(obs,
                                            state=lstm_states,
                                            episode_start=episode_starts,
                                            deterministic=True)

        obs, rewards, dones, info = eval_env.step(action)

        episode_starts = dones
        eval_env.render()
        time.sleep(0.1)

        if dones:
            time.sleep(1)
            obs = eval_env.reset()


class SaveCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, verbose: int = 1):
        super(SaveCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f'Saving model at step {self.num_timesteps}')
            self.model.save('model')

        return True


class MiniHackExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(MiniHackExtractor, self).__init__(observation_space,
                                                features_dim=1)
        self.embedding_size = 32

        self.embed = nn.Embedding(128, self.embedding_size)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():

            if key == 'chars':
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(
                    nn.Conv2d(self.embedding_size, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                total_concat_size += 16 * 21 * 79
            elif key == 'chars_crop':
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                    nn.Conv2d(self.embedding_size, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                total_concat_size += 16 * 9 * 9

        self.extractors = nn.ModuleDict(extractors)

        self.fc1 = nn.Linear(total_concat_size, 512)
        self.fc2 = nn.Linear(512, 512)

        # Update the features dim manually
        self._features_dim = 512

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # Add a channel to the 2d observation
            if len(observations[key].shape) < 4:
                observations[key] = observations[key]
            x = self.embed(observations[key].to(torch.int32))
            x = x.view(-1, self.embedding_size, x.shape[1], x.shape[2])
            x = extractor(x)

            encoded_tensor_list.append(x)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        x = torch.cat(encoded_tensor_list, dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def make_env(env_id, rank, seed=0, args={}):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id, **args)
        # env._max_episode_steps = 30
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init

def go_right_reward(env, prev, curr, action):
    # Get the location of the @
    prev_loc = np.where(curr == ord('@'))
    # Reward the player for going right
    

def make_dummy_env(env_id, num_envs):
    args = {
        'observation_keys': ['chars', 'chars_crop'],
        'penalty_time': -0.005,
        'penalty_step': -0.1
    }

    # Create the vectorized environment
    env = DummyVecEnv(
        [make_env(env_id, i, args=args) for i in range(num_envs)])
    return env


if __name__ == "__main__":
    env_id = "MiniHack-CorridorBattle-v0"
    n_envs = 12
    env = make_dummy_env(env_id, n_envs)

    # wrap env with a VecMonitor
    env = VecMonitor(env)

    model = RecurrentPPO("MultiInputLstmPolicy",
                         env,
                         verbose=1,
                         learning_rate=0.0002,
                         n_steps=128,
                         batch_size=128,
                         n_epochs=10,
                         gamma=0.99,
                         gae_lambda=0.95,
                         clip_range=0.2,
                         clip_range_vf=None,
                         normalize_advantage=True,
                         ent_coef=0.000001,
                         vf_coef=0.5,
                         max_grad_norm=40,
                         policy_kwargs=dict(
                             features_extractor_class=MiniHackExtractor,
                             ortho_init=True,
                             optimizer_class=RMSprop,
                             optimizer_kwargs=dict(alpha=0.99, eps=0.000001),
                             activation_fn=nn.ReLU,
                             enable_critic_lstm=False,
                             lstm_hidden_size=128,
                             net_arch=[512]))

    save_callback = SaveCallback(max(5000 // n_envs, 1))

    try:
        model.learn(total_timesteps=int(100_000), callback=save_callback)
    except KeyboardInterrupt:
        pass

    eval_env = make_dummy_env(env_id, 1)

    evaluate_lstm(model, eval_env)