# DQN

William Hill: 2115261  
Ireton Liu: 2089889  
Andrew Boyley: 2090244

## Structure

The structure of our submission is as follows:

- `dqn/` contains the source code for the agent, model, replay buffer and wrappers
- `inference_recordings/` contains the recording of our agent playing Pong after having completed training
- `DQN.ipynb` is the notebook used to train on Google Colab
- `episode-loss.png` and `episode-reward.png` are the graphs of loss and rewards over episodes, respectively, during training
- `group_members.txt` contains the names and student numbers of members of our group
- `infer_atari.py` is the script used to play a game with a previously-trained agent
- `model.pt` is the saved Policy Network of our agent after training. This is loaded when we do inference.
- `report.pdf` is the write-up containing a discussion of the training algorithm, DQN network architecture, and hyper-parameters
- `train_atari.py` is the script used to train our agent, and saves the final Policy Network to `model.pt`
