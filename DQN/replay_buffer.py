import numpy as np


class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """

        # will use a circular array implementation
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    # override the len() function for this object
    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        # create the tuple to add to the buffer
        data = (state, action, reward, next_state, done)

        # add the tuple to the buffer, overwriting the oldest one if the buffer is full, using a circular array structure
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        # get the transition tuple at each of the indices and split them up into the component quantities
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # return arrays of the component quantities
        return (
            states,
            np.array(actions),
            np.array(rewards),
            next_states,
            np.array(dones),
        )

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        # randomly sample batch_size indices from the buffer (within the bounds of the buffer size)
        indices = np.random.randint(0, len(self._storage) - 1, size=batch_size)

        # get those indices and return them
        return self._encode_sample(indices)
