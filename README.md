# DQN

To train the DQN agent:

- `cd` into the **sb3** folder.
- run `python3 sb3_dqn_minihack.py`
- This will generate a **model.zip** file in the same diretory.

To evaluate and visualise the model in the environment:

- ensure that there is a **model.zip** file already
- run `python3 sb3_dqn_minihack.py --eval`

To change the hyperparameters (for training):

- go to line 351 in the file **sb3_dqn_minihack.py**
- change the relevant values

**_Note: the current model.zip file is our current (best) agent._**
