import gym
import minihack
import time
import argparse
import os
import numpy as np

from typing import Any, Dict

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import Video
from torch.optim import RMSprop
from minihack import RewardManager
from generate_level import MazeGen
import torch
from torch import nn


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
            x = self.embed(observations[key].to(torch.int32))
            x = x.view(-1, self.embedding_size, x.shape[1], x.shape[2])
            x = extractor(x)

            encoded_tensor_list.append(x)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        x = torch.cat(encoded_tensor_list, dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def make_env(generator, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = generator.generate()

        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


class VideoRecorderCallback(BaseCallback):

    def __init__(self,
                 eval_env: gym.Env,
                 render_freq: int,
                 n_eval_episodes: int = 1,
                 deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any],
                             _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            # https://wrongsideofmemphis.com/2010/03/01/store-standard-output-on-a-variable-in-python/
            self.logger.record(
                "trajectory/video",
                Video(torch.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True

def go_right_bonus(env, prev, action, curr):
    # Get the x coord of the @
    try:
        x = np.where(curr[1] == ord('@'))[1][0]
        
        # Reward the player for moving towards the stairs
        return x * 0.001
    except:
        return 0

def make_dummy_env(num_envs, cls=DummyVecEnv):

    reward_manager = RewardManager()
    reward_manager.add_wield_event("death", reward=5)
    reward_manager.add_wield_event("cold", reward=5)
    reward_manager.add_wield_event("frost horn", reward=5)
    reward_manager.add_wear_event("levitation", reward=5)
    reward_manager.add_wear_event("levitation boots", reward=5)
    reward_manager.add_kill_event("Minotaur", reward=10, terminal_required=True)
    reward_manager.add_custom_reward_fn(go_right_bonus)

    args = dict(observation_keys=['chars', 'chars_crop'],
                max_episode_steps=1000,
                penalty_mode='linear',
                reward_lose=-1.0,
                penalty_time=-0.005,
                penalty_step=-0.1,
                reward_manager=reward_manager)

    maze_gen = MazeGen(args)

    # Create the vectorized environment
    env = cls([make_env(maze_gen, i) for i in range(num_envs)])
    return env


def init_argparse() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Print or check SHA1 (160-bit) checksums.")

    parser.add_argument("-e",
                        "--env_id",
                        type=str,
                        default="MiniHack-CorridorBattle-v0",
                        help="Environment ID")
    parser.add_argument("-n",
                        "--n_envs",
                        type=int,
                        default=16,
                        help="Number of environments")
    parser.add_argument("-s",
                        "--seed",
                        type=int,
                        default=0,
                        help="Random seed")
    # Parse how long to train for
    parser.add_argument('-l',
                        '--length',
                        type=int,
                        default=200_000,
                        help='How long to train for')

    parser.add_argument("-t",
                        "--eval",
                        default=False,
                        action="store_true",
                        help="Evaluation mode")
    
    parser.add_argument('-f','--fresh', default=False, action='store_true', help='Start fresh')

    parser.add_argument("-r", "--save_frequency", type=int, default=5000)

    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()

    if args.eval:
        eval_env = make_dummy_env(1, cls=DummyVecEnv)
        model = RecurrentPPO.load(
            "/home/andrew/Documents/minihack-the-planet/sb3/model.zip")
        evaluate_policy(
            model,
            eval_env,
            render=True,
            callback=lambda _locals, _globals: time.sleep(1),
            # callback=grab_screens,
            n_eval_episodes=10,
            deterministic=True,
        )
        exit()

    env = make_dummy_env(num_envs=args.n_envs, cls=SubprocVecEnv)

    # wrap env with a VecMonitor
    env = VecMonitor(env)

    mode = None

    # Check if the model.zip file exists in the current directory
    if os.path.exists("model.zip") and not args.fresh:
        print("Loading model.zip")
        model = RecurrentPPO.load("model")
        # Add the env to the model
        model.set_env(env)
    else :
        model = RecurrentPPO("MultiInputLstmPolicy",
                            env,
                            verbose=1,
                            tensorboard_log="./minihack_tensorboard/",
                            learning_rate=0.0002,
                            n_steps=512,
                            batch_size=256,
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
                                ortho_init=False,
                                optimizer_class=RMSprop,
                                optimizer_kwargs=dict(alpha=0.99, eps=0.000001),
                                activation_fn=nn.ReLU,
                                enable_critic_lstm=True,
                                lstm_hidden_size=128,
                                net_arch=[512],
                            ))

    save_callback = SaveCallback(max(args.save_frequency // args.n_envs, 1))
    eval_env = make_dummy_env(1)
    # video_recorder = VideoRecorderCallback(eval_env, render_freq=5000)

    try:
        model.learn(total_timesteps=args.length,
                    tb_log_name='ppo-lstm',
                    reset_num_timesteps=False,
                    callback=[save_callback])
    except KeyboardInterrupt:
        pass

    evaluate_policy(
        model,
        eval_env,
        render=True,
        callback=lambda _locals, _globals: time.sleep(1),
        n_eval_episodes=1,
        deterministic=True,
    )
