import imageio
from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np

env_id = "LunarLander-v2"

# Parallel environments
env = make_vec_env(env_id, n_envs=16)

# Define learning rate schedule lambda function
lr = lambda progress_remaining: progress_remaining * 0.00083

# model = A2C("MlpPolicy",
#             env,
#             verbose=1,
#             n_steps=5,
#             learning_rate=lr,
#             gamma=0.995,
#             ent_coef=0.00001)

model = PPO("MlpPolicy",
            env,
            verbose=1,
            n_steps=1024,
            batch_size=64,
            gamma=0.999,
            n_epochs=4,
            gae_lambda=0.98,
            ent_coef=0.01)
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

imageio.mimsave("lander_ppo.gif",
                [np.array(img) for i, img in enumerate(images) if i % 2 == 0],
                fps=29)
