from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from torch import nn
import gym
import time


ENV_ID = 'CartPole-v1'

class SaveCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, n_envs: int, verbose: int = 1):
        super(SaveCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.n_envs = n_envs

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            print(f'Saving model at step {self.num_timesteps}')
            self.model.save('model')

        return True

def evaluate_lstm(model, env_id):
    num_envs = 1
    env = make_vec_env(env_id, n_envs=num_envs, vec_env_cls=DummyVecEnv)
    obs = env.reset()

    # cell and hidden state of the LSTM
    lstm_states = None
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs, ), dtype=bool)
    for _ in range(2000):
        action, lstm_states = model.predict(obs,
                                            state=lstm_states,
                                            episode_start=episode_starts,
                                            deterministic=True)

        obs, rewards, dones, info = env.step(action)

        episode_starts = dones
        env.render()

        if dones:
            time.sleep(1)
            obs = env.reset()


class MaskVelocityWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to mask velocity terms in
    observations. The intention is the make the MDP partially observable.
    Adapted from https://github.com/LiuWenlin595/FinalProject.
    :param env: Gym environment
    """

    # Supported envs
    velocity_indices = {
        "CartPole-v1": np.array([1, 3]),
        "MountainCar-v0": np.array([1]),
        "MountainCarContinuous-v0": np.array([1]),
        "Pendulum-v1": np.array([2]),
        "LunarLander-v2": np.array([2, 3, 5]),
        "LunarLanderContinuous-v2": np.array([2, 3, 5]),
    }

    def __init__(self, env: gym.Env):
        super().__init__(env)

        env_id: str = env.unwrapped.spec.id
        # By default no masking
        self.mask = np.ones_like((env.observation_space.sample()))
        try:
            # Mask velocity
            self.mask[self.velocity_indices[env_id]] = 0.0
        except KeyError:
            raise NotImplementedError(
                f"Velocity masking not implemented for {env_id}")

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation * self.mask


env = make_vec_env(ENV_ID,
                   n_envs=12,
                   wrapper_class=MaskVelocityWrapper)

env = VecMonitor(env)

model = RecurrentPPO(env=env,
                     policy="MlpLstmPolicy",
                     verbose=1,
                     n_steps=256,
                     vf_coef=0.0004859761059684678,
                     gamma=0.9969446008249913,
                     ent_coef=0.0001663564886802498,
                     gae_lambda=0.9986504875171683,
                     learning_rate=0.01136664591360294,
                     max_grad_norm=4.054230220677884,
                     n_epochs=10,
                     batch_size=128,
                     clip_range=0.1,
                     policy_kwargs=dict(net_arch=[dict(pi=[64], vf=[64])],
                                        lstm_hidden_size=64,
                                        activation_fn=nn.Tanh,
                                        ortho_init=False,
                                        enable_critic_lstm=False))

model.learn(total_timesteps=100_000, progress_bar=True)

evaluate_lstm(model, ENV_ID)