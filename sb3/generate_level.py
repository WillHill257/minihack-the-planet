import gym
import minihack
from minihack import MiniHackSkill
from nle import nethack


class MazeGen:
    def __init__(self, args, evaluation=False):

        self.only_evaluation = evaluation
        self.args = args
        MOVE_ACTIONS = tuple(nethack.CompassDirection)
        self.ALL_ACTIONS = MOVE_ACTIONS + (
            nethack.Command.PICKUP,
            nethack.Command.QUAFF,
            nethack.Command.ZAP,
            nethack.Command.WEAR,
            nethack.Command.FIRE,
            nethack.Command.APPLY,
            nethack.Command.RUSH,
        )

    def get_des(self):
        return f"""
# Inspired from Baalzebub level of NetHack
MAZE:"mylevel",' '
FLAGS: noteleport,corrmaze
GEOMETRY:right,center

MAP
-------------------------------------------------
|                   ----               ----
|          ----     |     -----------  |
| ------      |  ---------|.........|--|
| |....|  -------|.LLLLL..-----------------
---....|--|........LLLLL..................|----
+.........+........LLLLL......................|
---....|--|........LLLLL..................|----
| |....|  -------|.LLLLL..-----------------
| ------      |  ---------|.........|--|
|          ----     |     -----------  |
|                   ----               ----
-------------------------------------------------
ENDMAP

STAIR:levregion(01,00,15,20),levregion(15,1,70,16),up
# BRANCH:levregion(01,00,15,20),levregion(15,1,70,16)
# BRANCH:(1,6,1,6),(0,6,0,6)
{"BRANCH:levregion(01,00,15,20),levregion(15,1,70,16)" if self.only_evaluation else "BRANCH:(1,6,1,6),(0,6,0,6)" }
REGION:(0,0,45,20),lit,"ordinary"

$entry_room = selection:fillrect (3,4,6,8)

MAZEWALK:(00,06),west
STAIR:(44,06),down
DOOR:closed,(00,06)
DOOR:closed,(10,06)

# One of this object can help cross the lava
IF [50%] {{
    # levitate
    IF [33%] {{
        # potion of levitation
        OBJECT:('!',"levitation"),rndcoord($entry_room),blessed
    }} ELSE {{
        IF [50%] {{
            # ring of levitation
            OBJECT:('=',"levitation"),rndcoord($entry_room),blessed
        }} ELSE {{
            # levitation boots
            OBJECT:('[',"levitation boots"),rndcoord($entry_room),blessed
        }}
    }}
}} ELSE {{
    # freeze
    IF [50%] {{
        # wand of cold
        OBJECT:('/',"cold"),rndcoord($entry_room),blessed
    }} ELSE {{
        # frost horn
        OBJECT:('(',"frost horn"),rndcoord($entry_room),blessed
    }}
}}
# Wand of death
OBJECT:('/',"death"),rndcoord($entry_room),blessed


# The fellow in residence
MONSTER:('H',"Minotaur"),(34,06),asleep
"""

    def generate(self, start_col=1):
        self.start_col = start_col
        env = MiniHackSkill(
            des_file=self.get_des(), actions=self.ALL_ACTIONS, **self.args
        )
        return env


if __name__ == "__main__":
    args = {
        "observation_keys": ["chars", "chars_crop"],
        "penalty_mode": "linear",
        "penalty_time": -0.001,
        "penalty_step": -0.01,
    }
    maze_gen = MazeGen(args)
    maze = maze_gen.generate()
    maze.reset()
    maze.render()
