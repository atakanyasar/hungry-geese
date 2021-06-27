import numpy as np
import os, random, copy, pickle
from time import time
from kaggle_environments import make, evaluate, envs
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Action
from utils import num
from greedy_agent import greedy_agent
from random_agent import random_agent
from submission03 import smart_agent
from src.reinforced_agent import reinforced_agent, get_state
from src.model import Q
from src.model_greedy import greedyQ
from plot import plot

import src.reinforced_agent
import src.model
import torch


outp = './outputs/'
log = './logs/'
data = outp

def train(name):
      
      agents = [None, reinforced_agent, reinforced_agent, reinforced_agent]

      env = make('hungry_geese', debug=True, configuration={'episodeSteps':200})

      trainer = env.train(agents)

      iterations = 10

      prev_obs = trainer.reset()
      obs = trainer.reset()
      r = 0
      done = 0

      eps_range = (0.2, 0.3)
      def eps(game):
            return eps_range[1] - (eps_range[1] - eps_range[0]) * game / iterations 

      begin = time()

      for game in range(1, iterations+1):

            env.render()
            prev_move = None

            while not done:
                  explore = (np.random.rand() < eps(game))
                  if explore and game > 1:
                        action = reinforced_agent(copy.deepcopy(obs), env.configuration, explore=True)
                        # print(f"random action:{action}")
                  else:
                        action = reinforced_agent(copy.deepcopy(obs), env.configuration)
                        # print(f"action:{action}")
                  
                  next_obs, reward, done, info = trainer.step(action)

                  if reward:
                        if reward == 101:
                              reward = 100 + 800 / len(next_obs['geese'][next_obs['index']])
                        else:
                              reward = 100
                        
                        if done and obs['step'] >= 150:
                              reward += 20000
                  # else:
                  #       reward = -2000

                  r += reward
                  
                  move = torch.zeros(4)
                  move[num[action]] = 1

                  # if eps(game) > 0. or not explore:
                  Q.remember(
                        get_state(copy.deepcopy(obs), num[prev_move]),
                        move,
                        reward/5e4,
                        done,
                        get_state(copy.deepcopy(next_obs), num[action]) if not done else torch.zeros((1, 46, 7, 11))
                  )

                  prev_obs = copy.deepcopy(obs)
                  obs = next_obs
                  prev_move = action

            print(f"game {game}: reward:{r/5e4} steps:{obs['step']} expected time:{(time()-begin)/game * (iterations-game)/60} minutes")
            print(r, file=open(log+"log.txt", "a"))
            print(obs['step'], file=open(log+"surviving.txt", "a"))

            if game % 2 == 0:
                  plot()

            obs = trainer.reset()
            r = 0
            done = 0

      Q.turnoff()

if __name__ == '__main__':
      train(1)
      