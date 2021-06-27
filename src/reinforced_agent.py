import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration, Action, Observation, row_col
from copy import deepcopy
from torch import Tensor
import torch
#utils here

from utils import rows, columns, num, make_grid, adjust, BFS, reverseBFS, pair, go
# from src.model import Q
from src.model_greedy import Q

class ReinforcedAgent:
      def __init__(self, config):
            self.max_eps = config['episodeSteps']
            self.time_out = config['actTimeout']
            self.columns = config['columns']
            self.rows = config['rows']
            self.hunger_rate = config['hunger_rate']

            self.move = None
            self.obses = []
            self.train = False
            

      def __call__(self, obs, explore=False):
            self.obses.append(obs)
            self.train = (obs['index'] == 0)
            prev_obs = self.obses[-2] if len(self.obses) > 1 else obs

            self.model_input = self.state(obs, num[self.move])
            
            self.move = self.choose(explore)

            return self.move.name
      
      def state(self, obs, prev_move):
            return self.make_input(deepcopy(obs), prev_move)

      def choose(self, explore=False):
            if explore:
                  probability = np.random.rand() * self.possible_moves()
            else:
                  probability = Q.model(self.model_input).cpu() #* self.possible_moves()
            
            # if self.train:
            #       print(f"                      {probability.detach().numpy()}")
            
            move = int(torch.argmax(probability))
            for x, y in enumerate(Action):
                  if x == move:
                        return y

      def possible_moves(self):
            return torch.tensor([self.move == None or Action.opposite(self.move) != i for i in Action])

      def make_input(self, obs, prev_move):
            remaining_time = obs['remainingOverageTime']
            step = obs['step']
            geese = obs['geese']
            foods = obs['food']
            index = obs['index']
            
            grid = make_grid(geese)
            grid = adjust(grid, geese, index, foods)

            geese = [i for i in geese if len(i) > 0]

            for i in range(len(geese)):
                  for j in range(len(geese[i])):
                        geese[i][j] = pair(geese[i][j])
            foods = [pair(i) for i in foods]
            lenghts = [len(i) for i in geese]
            heads = [goose[0] for goose in geese]


            bfs_foods = np.array([BFS(food, grid, let=heads) for food in foods])
            nearest = np.array([np.minimum(bfs_foods[0][goose[0]], bfs_foods[1][goose[0]]) for goose in geese])
            

            for g, goose in enumerate(geese):
                  for j in range(len(goose)):
                        if len(goose) - j < nearest[g]:
                              grid[goose[j]] = len(goose) - j
                        else:
                              grid[goose[j]] = len(goose) - j + 1
            
            bfs_geese = np.array([BFS(goose[0], grid, from_head=True) for goose in geese])
            bfs_tail = np.array([BFS(goose[-1], grid, v=grid[goose[-1]]-1, let=heads, from_head=True) for goose in geese])

            ########################################################################################################################################
            
            ##### GREEDY MODEL #####

            greedy_prev = np.zeros((4))
            greedy_adj = np.zeros((4))
            greedy_food = np.zeros((8))
            greedy_others = np.zeros((8))

            if prev_move != None:
                  greedy_prev[prev_move] = 1
      
            greedy_adj = np.array(
                  [grid[go(heads[0], a)] <= 1 for a in Action]
            )
            
            greedy_food = np.array(
                  [[food_i[go(heads[0], a)] for food_i in bfs_foods] for a in Action]
            ).reshape(-1) / 10

            greedy_others = np.array(
                  [[min([food_i[go(head, a)] for head in heads[1:]]) for food_i in bfs_foods] for a in Action]
            ).reshape(-1) / 10

            
            greedy = np.concatenate([
                  greedy_prev,
                  greedy_adj,
                  greedy_food,
                  greedy_others
            ], axis=0).reshape(1, -1)

            assert greedy.shape == (1, 24)

            return Tensor(greedy)
            
            ##### GREEDY MODEL #####

            data_food = np.zeros((11, 7, 11))
            data_geese = np.zeros((6, 7, 11))
            data_head = np.zeros((12, 7, 11))
            data_tail = np.zeros((12, 7, 11))
            data_prev = np.zeros((4, 7, 11))
            data_len = np.zeros((1, 7, 11))

            for r in range(rows):
                  for c in range(columns):
                        data_geese[min(grid[r, c], 5)][r][c] = 1
            for g, goose in enumerate(bfs_geese):
                  for r in range(rows):
                        for c in range(columns):
                              data_head[6*min(g, 1)+min(goose[r, c], 5)][r][c] += 1
            for g, goose in enumerate(bfs_tail):
                  for r in range(rows):
                        for c in range(columns):
                              data_tail[6*min(g, 1)+min(goose[r, c], 5)][r][c] += 1
            
            for food in bfs_foods:
                  for r in range(rows):
                        for c in range(columns):
                              data_food[min(food[r, c], 10)][r][c] += 1
            if prev_move != None:
                  data_prev[prev_move][:] = 1
            data_len[:] = (len(geese[0])*hunger_rate-step%hunger_rate)/1e4

            model_input = np.concatenate([data_food, data_geese, data_head, data_tail, data_prev, data_len], axis=0)

            # try:
            #       import os
            #       os.mkdir('inputs')
            # except:
            #       pass

            # for i, inp in enumerate(model_input):
            #       with open('./version/inputs/input'+str(i)+'.txt', 'a') as f:
            #             print(f'step:{str(self.step)}', file=f)
            #             np.savetxt(f, inp, fmt='%.2f')
            
            assert(model_input.shape == (46, 7, 11))

            model_input = Tensor(model_input.reshape((1,)+model_input.shape))

            return model_input
      
                  
cache = {}


def get_state(obs, prev_move):
      return cache[obs['index']].state(obs, prev_move)

def reinforced_agent(obs, config, explore=False):
      if not obs['index'] in cache:
            cache[obs['index']] = ReinforcedAgent(config)
            
      return cache[obs['index']](obs, explore)


