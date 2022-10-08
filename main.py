import gym
import minihack

env = gym.make("MiniHack-Quest-Easy-v0",
   observation_keys=["glyphs","blstats","inv_glyphs"])
# actions[50] == PRAY
observation = env.reset() # each reset generates a new environment instance
observation, reward, terminated, info = env.step(1)  # move agent '@' north
env.render()