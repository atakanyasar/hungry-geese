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
# from src.model import Q
from src.model_greedy import Q
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

      iterations = 1000

      prev_obs = trainer.reset()
      obs = trainer.reset()
      r = 0
      done = 0

      eps_range = (0.1, 0.5)
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
                  else:
                        action = reinforced_agent(copy.deepcopy(obs), env.configuration)
                  
                  next_obs, reward, done, info = trainer.step(action)

                  if reward:
                        if reward == 101:
                              reward = 1
                        else:
                              reward = 0

                  r += reward
                  
                  move = torch.zeros(4)
                  move[num[action]] = 1

                  Q.remember(
                        get_state(copy.deepcopy(obs), num[prev_move]),
                        move,
                        reward/5,
                        done,
                        get_state(copy.deepcopy(next_obs), num[action]) if not done else torch.zeros((1, 24))
                  )

                  prev_obs = copy.deepcopy(obs)
                  obs = next_obs
                  prev_move = action

            print(f"game {game}: reward:{r/5} steps:{obs['step']} expected time:{(time()-begin)/game * (iterations-game)/60} minutes")
            print(r, file=open(log+"log.txt", "a"))
            print(obs['step'], file=open(log+"surviving.txt", "a"))

            # if game % 3 == 0:
            #       plot()

            obs = trainer.reset()
            r = 0
            done = 0

      Q.turnoff()

if __name__ == '__main__':
      train(1)
      