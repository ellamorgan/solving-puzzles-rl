import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import DataLoader

from networks import VAE_Model

'''
Deep Q Learning: uses two neural networks to learn and predict what 
action to take at every step.

One network is the 'Q network' or 'online network' is used to predict
what to do when the agent encounters a new state.
Takes in state and outputs Q values for the possible actions that
could be taken.

Experience Replay: this is what's used to learn and improve. This is
where the second target network comes into play.

A random set of experiences is retrieved from the agents memory.
For each experience, new Q values are calculated for each state and
action pair.

value = reward + discount_factor * target_network.predict(next_state)[argmax(online_network.predict(next_state))]

The neural network is fit to associate each state in the batch with
the new Q values calculated for the actions taken

'''

class DQN(nn.Module):
    '''
    DQN is a simple feedforward network that predicts the Q values
    '''
    def __init__(self, f, n_actions, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(f, hidden_size)
        self.linear2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class DoubleDQN:
    
    def __init__(self, args, device):

        self.memory = []
        self.max_memory_size = args['max_memory_size']
        self.online_network = DQN(args['fluents'], args['n_actions'], args['hidden_size'])
        self.target_network = DQN(args['fluents'], args['n_actions'], args['hidden_size'])
        self.vae = self.load_vae_network(args, device)
        self.experience_batch = args['experience_batch']
        self.batch_size = args['batch']
        self.n_actions = args['n_actions']

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        self.loss = nn.MSELoss()

        self.gamma = args['gamma']
        self.epsilon = args['epsilon']

        self.device = device
        self.usecuda = args['usecuda']
    

    def load_vae_network(self, args, device):
        '''
        Loads in the pretrained vae - the encoder portion of the vae is used to encode the states
        '''
        vae = VAE_Model(**args, device=device)
        vae.load_state_dict(torch.load(args['vae_model_path']))
        vae.eval()
        return vae
    

    def choose_action(self, state):
        '''
        Chooses the action with a probability of epsilon to explore (choose the action at random)
        and (1 - epsilon) probability of exploiting (choosing the action based on the online network)
        '''
        sample = random.random()

        if sample > self.epsilon:
            # Exploit
            with torch.no_grad():
                return self.online_network(state).max(1)[1].view(1, 1)
        else:
            # Explore
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def experience_replay(self):
        '''
        Samples transitions from memory and calculates Q update values
        '''
        if len(self.memory) < self.experience_batch:
            return

        # Samples the transitions to train on from memory
        experience_samples = random.sample(self.memory, self.experience_batch)
        new_q_values = []
        states = []

        for transition in experience_samples:
            prev, sucs, action, reward, done = transition
            states.append(prev)

            # Get current Q value predictions - will update with the desired Q value
            new_q_value = self.online_network(prev)

            if done:
                # Q value is 0 if at terminal state
                q_update = reward

            else:
                # Select action from online network
                online_network_action = np.argmax(self.online_network(sucs).detach().numpy())

                # Evaluate action through target network, calculate Q update
                # This becomes the value we want the network to predict
                q_update = reward + self.gamma * self.target_network(sucs)[0, online_network_action]

            # Update the q values
            new_q_value[0, action] = q_update.float()
            new_q_values.append(new_q_value)
        
        states = torch.stack(states)
        new_q_values = torch.stack(new_q_values)

        # Training step with new q values
        self.training_update(states, new_q_values)
    

    def training_update(self, states, target_q_values):
        '''
        Performs a single optimization step over the new Q updates evaluated after sampling from memory
        '''
        training_data = DataLoader([[states, target_q_values]], batch_size=self.batch_size, pin_memory=self.usecuda)
        training_loss = 0

        # Performs a single optimization step over the newly calculated Q values
        for data, target in training_data:
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            out = self.online_network(data)
            loss_val = self.loss(out, target)
            loss_val.backward(retain_graph=True)
            self.optimizer.step()
            training_loss += loss_val.item()
        training_loss /= self.experience_batch

        return training_loss
    

    def add_to_memory(self, prev, sucs, action, reward, done):
        '''
        Adds a transition to memory
        '''
        self.memory.append([prev, sucs, action, reward, done])
        if len(self.memory) == self.max_memory_size + 1:
            self.memory = self.memory[1:]
    

    def match_networks(self):
        '''
        The target network is updated with the online networks weights after a certain number of epochs
        '''
        self.target_network.load_state_dict(self.online_network.state_dict())
    

    def encode_state(self, state):
        '''
        Encodes an image representation of a state as a discrete representation
        '''
        return self.vae.encode(state)