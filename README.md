# Minihack The Planet

## Structure

This branch (`main`) does not contain any implementations, but rather this README that explains how the rest of the repository is structured.

We have trained agents using two different algorithms, each of which can be found in its own branch:

- a DQN agent can be found in the `sb3_dqn` branch
- a RecurrentPPO agent can be found in the `sb3_recurrentppo`

Each branch contains a README with the relevant instructions for training and watching the agents.

## Setup

**Ensure that you are using Gym version 0.21.0 and the latest versions of Nethack (NLE) and Minihack.**

If the installation of NLE or Minihack fail on the latest version of Ubuntu, the source code may need to be edited and built from scratch. Instructions for what to change are included below.

### Nethack Learning Environment (NLE)

1. Clone NLE

```
git clone https://github.com/facebookresearch/nle --recursive
```

2. Install brew

For Linux,

```
sudo apt-get install -y build-essential autoconf libtool pkg-config \
    python3-dev python3-pip python3-numpy git flex bison libbz2-dev
```

For Mac,

```
brew install cmake
```

3. Setup virtual environment

```
conda create -y -n nle python=3.8
```

4. Make NLE

```
pip install -e ".[all]"
```

### Minihack

1. Clone Minihack

```
git clone https://github.com/facebookresearch/minihack
```

2. Change things around

   In `base.py`:

   1. Replace all `self.env` with `self.nethack`

   In `agent/common/envs/wrapper.py`

   1. Replace all `env._actions` with `.actions`
