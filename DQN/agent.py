from gym import spaces
import numpy as np

from DQN.model import DQN
from DQN.replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class DQNAgent:

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
        device=device,
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        # initialise the agent's attributes, and create the two DQNs
        self.memory = replay_buffer
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.gamma = gamma
        self.device = device
        self.policy_network = DQN(observation_space,
                                  action_space).to(self.device)
        self.target_network = DQN(observation_space,
                                  action_space).to(self.device)
        self.update_target_network(
        )  # the two networks are randomly initialised -> this ensures we start from a consistent view
        self.target_network.eval(
        )  # we always want the target network in eval mode because we never train on it, only use to get the target values
        # use an ADAM optimiser
        self.optimiser = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=lr)  # bind the optimiser to the policy network's parameters

        self.q_values = None
        print(f'device: {self.device}')

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        #   Sample the minibatch from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size)

        # normalise the pixel values to [0, 1]

        # states = np.array(states).reshape(-1, 4, 21,79) / 5991.0
        # next_states = np.array(next_states).reshape(-1, 4, 21,79) / 5991.0

        # create tensors from the numpy arrays for use with pytorch
        # states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        # next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # get the output values from the target network
        # no grad since we do not want to do the update yet
        with torch.no_grad():
            if self.use_double_dqn:
                # get the best action from the policy network. max(1) since we greedily choose the action
                _, max_next_action = self.policy_network(next_states).max(1)

                # get the value of the corresponding action from the target network
                max_next_q_values = (self.target_network(next_states).gather(
                    1, max_next_action.unsqueeze(1)).squeeze())
            else:
                # if don't use a "target" network, just use all the values from the "current" network
                next_q_values = self.target_network(next_states)
                max_next_q_values, _ = next_q_values.max(1)

            # compute the target values using the actions decided on above (i.e. td-target)
            target_q_values = rewards + (
                1 - dones) * self.gamma * max_next_q_values

        # compute the values of the output nodes of the network
        input_q_values = self.policy_network(states)

        # only take the q-values corresponding to the nodes of the actions we took
        input_q_values = input_q_values.gather(1,
                                               actions.unsqueeze(1)).squeeze()

        # compute the td loss -> this is more of a constructor than a loss function, the loss is calculated during backprop
        loss = F.smooth_l1_loss(input_q_values, target_q_values)

        # don't want old gradients interfering, so initialise them to zero
        self.optimiser.zero_grad()

        # compute the gradients (backprop through entire network)
        # updates the .grad attribute of the weights, which are stored in the tensors themselves (input_q_values)
        loss.backward()

        # update the weights
        # uses the .grad attribute of the weights
        # https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step
        self.optimiser.step()

        del states
        del next_states
        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, state):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """

        state = [state]
        # do not accumulate gradients here, since we are simply evaluating, and not training
        with torch.no_grad():
            # greedily choose the best action
            self.q_values = self.policy_network(state)
            _, action = self.q_values.max(1)
            return action.item()

    def save_model(self, dir, episode, t):
        """
        save the model to the specified dir
        """

        torch.save(
            {
                'episode': episode,
                't': t,
                'model_state_dict': self.policy_network.state_dict(),
                'optimizer_state_dict': self.optimiser.state_dict(),
                'storage': self.memory._storage,
                'next_idx': self.memory._next_idx,
            }, dir)

    def load_model(self, dir):
        """
        load the policy network from the specified dir
        """
        return torch.load(dir)
