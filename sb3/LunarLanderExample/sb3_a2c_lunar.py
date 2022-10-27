import imageio
from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np

# env_id = "LunarLander-v2"
env_id = "CartPole-v1"

# Parallel environments
env = make_vec_env(env_id, n_envs=16)

# Define learning rate schedule lambda function
lr = lambda progress_remaining: progress_remaining * 0.00083

model = A2C("MlpPolicy",
            env,
            verbose=1,
            ent_coef=0.0)


try:
    model.learn(total_timesteps=1e6)
except:
    pass

images = []
env = make_vec_env(env_id, n_envs=1)
obs = env.reset()
img = env.render(mode="rgb_array")

for i in range(350):
    images.append(img)

    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)

    img = env.render(mode="rgb_array")
    env.render()

    if done:
        env.reset()

imageio.mimsave("cartpole_a2c.gif",
                [np.array(img) for i, img in enumerate(images) if i % 2 == 0],
                fps=29)
