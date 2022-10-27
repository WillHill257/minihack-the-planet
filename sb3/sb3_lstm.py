from curses import wrapper
import numpy as np

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
import os

from torch import nn
import gym


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


class SaveCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int,n_envs:int, verbose: int = 1):
        super(SaveCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.n_envs = n_envs

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            print(f'Saving model at step {self.num_timesteps}')
            self.model.save('model')

        return True


def evaluate(model, env_id):
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
            obs = env.reset()


if __name__ == "__main__":
    n_envs = 4

    env_id = 'LunarLander-v2'
    env = make_vec_env(env_id,
                       n_envs=n_envs,
                       seed=0,
                       wrapper_class=MaskVelocityWrapper,
                       vec_env_cls=SubprocVecEnv)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=16,
        ent_coef=0.01,
        gamma=0.999,
        gae_lambda=0.98,
        n_epochs=4,
        policy_kwargs=dict(ortho_init=False,
                           activation_fn=nn.ReLU,
                           lstm_hidden_size=8,
                           enable_critic_lstm=True,
                           net_arch=[dict(pi=[64], vf=[64])]),
        verbose=1,
    )

    callback = SaveCallback(check_freq=5000, n_envs=n_envs)

    try:

        model.learn(total_timesteps=int(5e6),
                    callback=callback,
                    progress_bar=True)
    except:
        print("Training interrupted. Moving to evaluation.")

    evaluate(model, env_id)