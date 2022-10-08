# minihack-the-planet

## Setup

### Nethack Learning Environment

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