import gym
import minihack
from nle import nethack

MOVE_ACTIONS = tuple(nethack.CompassDirection) # h,j,k,l,y,u,b,n
ALL_ACTIONS = MOVE_ACTIONS + (
    nethack.Command.ZAP,  # z
    nethack.Command.PRAY, # M-p
    nethack.Command.APPLY, # a
    nethack.Command.PICKUP, # ,

    nethack.Command.WEAR, # W

    nethack.Command.FIRE, # f
    nethack.Command.RUSH, # g 

)

env = gym.make(
    "MiniHack-MazeExplore-Easy-v0",
    observation_keys=["glyphs", "blstats", "inv_glyphs", "inv_letters"],
    actions=ALL_ACTIONS,
)
# actions[50] == PRAY
observation = env.reset() # each reset generates a new environment instance
observation, reward, terminated, info = env.step(1)  # move agent '@' north
env.render()