import numpy as np
from queue import Queue
from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration, Action, Observation, row_col
#utils here
from utils import rows, columns, make_grid, adjust, BFS, pair, will_grow

class SmartAgent:
      def __init__(self, config):
            self.max_eps = config['episodeSteps']
            self.time_out = config['actTimeout']
            self.columns = config['columns']
            self.rows = config['rows']
            self.hunger_rate = config['hunger_rate']

            self.move = None
            

      def __call__(self, obs):
            print(obs)
            self.remaining_time = obs['remainingOverageTime']
            self.step = obs['step']
            self.geese = obs['geese']
            self.foods = obs['food']
            self.index = obs['index']
            
            self.grid = make_grid(self.geese)
            self.grid = adjust(self.grid, self.geese, self.index, self.foods)

            self.geese = [i for i in self.geese if len(i) > 0]
            for i in range(len(self.geese)):
                  for j in range(len(self.geese[i])):
                        self.geese[i][j] = pair(self.geese[i][j])
            self.foods = [pair(i) for i in self.foods]
            self.lenghts = [len(i) for i in self.geese]

            for g in self.geese:
                  if will_grow(g, self.foods):
                        for j in range(len(g)):
                              self.grid[g[j]] = len(g) - j + 1
                  
                  else:
                        for j in range(len(g)):
                              self.grid[g[j]] = len(g) - j
            self.bfs_geese = [BFS(self.geese[i][0], self.grid.copy(), from_head=True) for i in range(1, len(self.geese))]
            # for val in self.bfs_geese:
            #       self.grid[val <= 1] = 100
            
            self.move = self.choose(self.possible_moves())
            return self.move.name

      def possible_moves(self):
            return [i for i in Action if self.move == None or Action.opposite(self.move) != i]

      def choose(self, actions, r=3, c=5):

            def choose_by(val=None, evaluate=None):
                  probability = [1., 1., 1.]

                  if evaluate == None:
                        def evaluate(s):
                              return -val[s]

                  for i in range(3):
                        a = actions[i]
                        if a == Action.NORTH:
                              probability[i] = evaluate((r-1, c))
                        if a == Action.SOUTH:
                              probability[i] = evaluate((r+1, c))
                        if a == Action.EAST:
                              probability[i] = evaluate((r, c+1))
                        if a == Action.WEST:
                              probability[i] = evaluate((r, c-1))
                  # print("in here", probability, val)
                  return actions[int(np.argmax(probability))]
            
            #option1: go to food if it is closer
            bfs_food = [
                  BFS(self.foods[0], self.grid.copy(), let=[self.geese[i][0] for i in range(0, len(self.geese))]), 
                  BFS(self.foods[1], self.grid.copy(), let=[self.geese[i][0] for i in range(0, len(self.geese))]) 
                  ]
            
            if bfs_food[0][(3, 5)] > bfs_food[1][(3, 5)]:
                  bfs_food = [bfs_food[1], bfs_food[0]]
            
            ok = np.array([bfs_food[0][(3, 5)] < bfs_food[0][self.geese[i][0]] for i in range(1, len(self.geese))])
            if np.sum(ok) == len(ok) and ((np.argmax(self.lenghts) != 0 and self.lenghts[0] <= 12) or self.lenghts[0] < 6 or bfs_food[0][(3, 5)] <= 1):
                  return choose_by(bfs_food[0])
            
            ok = np.array([bfs_food[1][(3, 5)] < bfs_food[0][self.geese[i][0]] for i in range(1, len(self.geese))])
            if np.sum(ok) == len(ok) and ((np.argmax(self.lenghts) != 0 and self.lenghts[0] <= 12) or self.lenghts[0] < 6 or bfs_food[1][(3, 5)] <= 1):
                  return choose_by(bfs_food[1])
            

            #option2: try to survive
            def evaluate(s):
                  if self.grid[s] > 1:
                        return -100
                  v = BFS(s, self.grid.copy(), v=1)
                  for i in range(1, len(self.geese)):
                        if abs(s[0]-self.geese[i][0][0]) + abs(s[1]-self.geese[i][0][1]) == 0:
                              return -np.mean(v)-5
                        if abs(s[0]-self.geese[i][0][0]) + abs(s[1]-self.geese[i][0][1]) == 1:
                              return -np.mean(v)-4
                  return -np.mean(v)
            
            return choose_by(evaluate=evaluate)

                  
cache = {}

def smart_agent(obs, config):
      if not obs['index'] in cache:
            cache[obs['index']] = SmartAgent(config)
            
      return cache[obs['index']](obs)