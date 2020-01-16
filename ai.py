# AI for Self Driving Car
# Importing the libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Neural Network Class
class NeuralNetwork(nn.Module):

    def __init__(self, input_size, actions_num):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.actions_num = actions_num
        self.connection_one = nn.Linear(input_size, 30)
        self.connection_two = nn.Linear(30, actions_num)

    def forward(self, current_state):
        hidden = F.relu(self.connection_one(current_state))
        q_vals = self.connection_two(hidden)
        return q_vals

# Experience Replay Class -- Markov Decision Process & Long Term Memory Concepts
class MemoryReplay(object):

    def __init__(self, replays):
        self.replays = replays
        self.memory = []
    
    def push_memory(self, event):
        self.memory.append(event)
        if len(self.memory) > self.replays:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# Implementing DQN 
class DeepQ():
    def __init__(self, input_size, actions_num, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = NeuralNetwork(input_size, actions_num)
        self.memory = MemoryReplay(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze[0]
        self.last_action = 0
        self.last_reward = 0

    def get_action(self, current_state):
        modified_state = Variable(current_state, volatile = True) 
        probabilities = F.softmax(self.model(modified_state)*7) #Temperature = 7, higher prob of winning Q val 
        selected_action = probabilities.multinomial()
        return selected_action.data[0,0]
    
    def q_learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze[1]).squeeze[1]
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = batch_reward + self.gamma*next_outputs
        loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        loss.backward(retain_variables = True) 
        self.optimizer.step()

    def update(self, reward, signal):
        new_state = torch.Tensor(signal).float().unsqueeze(0)
        self.memory.push_memory((self.last_state, new_state, torch.LongTensor([int(self.last_action)])), torch.Tensor(self.last_reward))
        selected_action = self.get_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.q_learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = selected_action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return selected_action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")











