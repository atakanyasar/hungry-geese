import numpy as np
import os, pickle
import random
import torch
import torch.nn as nn
from collections import deque
from torch import Tensor
from torch import tensor
from time import time, sleep
import src.model_greedy

if torch.cuda.is_available():
      cuda = torch.device('cuda')
else:
      cuda = torch.device('cpu')

data_path = './outputs/'
model_path = data_path+'model.bin'
memory_path = data_path+'memory.bin'

test = []

if not 'outputs' in os.listdir():
      os.mkdir('outputs')
      
if not 'logs' in os.listdir():
      os.mkdir('logs')

greedy_model = src.model_greedy.Model()
greedy_model.load_state_dict(torch.load(data_path+'model_greedy.bin'))

class Model(nn.Module):
      input_shape = (46, 7, 11)
      output_shape = 4

      def __init__(self):
            super(Model, self).__init__()
            channels = 64

            

            self.conv1 = nn.Conv2d(self.input_shape[0], channels, (3, 3), bias=False, padding=1, padding_mode='circular')
            self.conv2 = nn.Conv2d(channels, channels, (3, 3), bias=False, padding=1, padding_mode='circular')
            self.lrelu = nn.LeakyReLU()
            self.relu = nn.ReLU()
            self.logsoftmax = nn.LogSoftmax(dim=1)
         
            self.flatten = nn.Flatten()
            self.ln1 = nn.Linear(channels*7*11, channels)
            self.ln2 = nn.Linear(channels, channels)
            self.ln3 = nn.Linear(channels, self.output_shape)

            self.layers = [
                  self.conv1,
                  nn.LeakyReLU(),
                  self.conv2,
                  nn.LeakyReLU(),
                  self.flatten,
                  self.ln1,
                  nn.LeakyReLU(),
                  self.ln2,
                  nn.LeakyReLU(),
                  self.ln3,
                  nn.LeakyReLU()
            ]

            self.criterion = nn.MSELoss()
            

      def forward(self, x):
            x = x.to(cuda)
            for l in self.layers:
                  if __name__ == '__main__' and l == self.layers[10]:
                        print("\n",l, l(x).data.cpu().detach().numpy())
                        test.append(l(x).data.cpu().detach().numpy())
                  try:
                        x = l(x)
                  except:
                        assert 0, f"something wrong with layer {l}"
            return x

      def gradientDescent(self, y_pred, y_true):
            # print(y_pred[0], y_true[0])
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
            self.optimizer.zero_grad()
            loss = self.criterion(y_pred, y_true)
            loss.backward()
            self.optimizer.step()

            # print(f"          loss={loss.item()}")


class DQN:
      def __init__(self, gamma=0.9, batch_size=256, freq=5, memory_limit=20000):
            self.model = Model()
            self.model.to(cuda)

            self.gamma = gamma
            self.memory = list()
            self.memory_limit = memory_limit
            self.batch_size = batch_size
      
            
            self.learn_freq = freq
            self.save_freq = 5000
            self.step = 0


      def remember(self, state, action, reward, done, next_state):
            self.step += 1
            
            self.memory.append([state.to(cuda), action.to(cuda), reward, int(done), next_state.to(cuda)])
            if done:
                  self.memory.append([state.to(cuda), action.to(cuda), reward, int(done), next_state.to(cuda)])
            
            while len(self.memory) > self.memory_limit:
                  self.memory.pop(0)
            
            size = min(self.batch_size, len(self.memory))
            if self.step % self.learn_freq == 0:
                  self.train(random.sample(self.memory, size))
            if self.step % (self.learn_freq*5) == 0:
                  self.train(self.memory[-size:])
            if self.step % self.save_freq == 0:
                  self.save()

      def train(self, memory, epochs=1):
            # print(f"    step={self.step} training started")
            
            for _ in range(epochs):
                  
                  state = torch.cat([sars[0] for sars in memory], axis=0).to(cuda)
                  action = torch.cat([sars[1] for sars in memory], axis=0).to(cuda).view(-1, 4)
                  reward = torch.tensor([sars[2] for sars in memory], device=cuda).view(-1, 1)
                  done = torch.tensor([sars[3] for sars in memory], device=cuda).view(-1, 1)
                  next_state = torch.cat([sars[4] for sars in memory], axis=0).to(cuda)

                  oldQ = self.model(state)
                  targetQ = (1-done)*torch.max(self.model(next_state).detach(), axis=1, keepdim=True).values
                  targetQ = (1-action)*oldQ + action*(reward + self.gamma*targetQ)
                  # print(oldQ[-1].data)
                  # print(targetQ[-1].data)
                  # print(oldQ[done.view(-1)==1][-1].data)
                  # print(targetQ[done.view(-1)==1][-1].data)
                  self.model.gradientDescent(oldQ, targetQ)
                  
            # print(f"    {time()-b} second")

      def save(self):
            while True:
                  try:
                        with open(model_path, 'wb') as f:
                              torch.save(self.model.state_dict(), f)
                        print("MODEL SAVED")
                  except:
                        print("Warning: model saving failed, retring to save")
                        sleep(0.1)
                        continue
                  break

            while True:
                  try:
                        with open(memory_path, 'wb') as f:
                              pickle.dump(self.memory, f)
                        print("MEMORY SAVED")
                  except:
                        print("Warning: memory saving failed, retring to save")
                        sleep(0.1)
                        continue
                  break
      def load(self, load_memory=False):
            if 'model.bin' in os.listdir(data_path):
                  while True:
                        try:
                              self.model.load_state_dict(torch.load(model_path))
                              print("MODEL LOADED")
                        except:
                              print("Warning: model loading failed, retring to load")
                              sleep(0.1)
                              continue
                        break

            if 'memory.bin' in os.listdir(data_path) and load_memory:
                  while True:
                        try:
                              with open(memory_path, 'rb') as f:
                                    self.memory = pickle.load(f)
                              print("MEMORY LOADED")
                        except:
                              print("Warning: memory loading failed, retring to load")
                              sleep(0.1)
                              continue
                        break
      def turnoff(self):
            self.save()

global Q
Q = DQN()
try:
      Q.model.load_state_dict(torch.load(model_path))
      print("MODEL LOADED")
except:
      print("NEW MODEL")

if __name__ == '__main__':
      print("model.py is working")
      inp = torch.zeros((1, 46, 7, 11))
      q = tensor(inp)
      # for i in range(41, 45):
      #       inp[0][i][:] = 1
      #       # q = torch.cat((q, inp), axis=0)
      #       p = Q.model(inp)
      #       inp[0][i][:] = 0
      #       # print(p)
      for r, c in [[2, 5], [3, 6], [4, 5], [3, 4]]:
            inp[0][12][r][c] = 1
            p = Q.model(inp)
            inp[0][12][r][c] = 0
            # print(p.data)
      # print(np.mean(np.abs(test[0]-test[1])))
