import numpy as np
from queue import Queue
from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration, Action, Observation, row_col
from utils import rows, columns, make_grid, adjust

class RandomAgent:
      def __init__(self, config):
            self.max_eps = config['episodeSteps']
            self.time_out = config['actTimeout']
            self.columns = config['columns']
            self.rows = config['rows']
            self.hunger_rate = config['hunger_rate']

            self.move = None
            

      def __call__(self, obs):
            self.remaining_time = obs['remainingOverageTime']
            self.step = obs['step']
            self.geese = obs['geese']
            self.foods = obs['food']
            self.index = obs['index']
            
            self.grid = make_grid(self.geese)
            self.grid = adjust(self.grid, self.geese, self.index, self.foods)
            
            self.risk = self.risk_map()
            self.reward = self.reward_map()

            actions = self.possible_moves()
            
            self.move = np.random.choice(actions)
            return self.move.name

      def possible_moves(self):
            return [i for i in Action if self.move == None or Action.opposite(self.move) != i]

      def risk_map(self):
            bfs = Queue()
            risk = np.zeros((self.rows, self.columns))

            for r in range(self.rows):
                  for c in range(self.columns):
                        if self.grid[r][c] != 0:
                              bfs.put((r, c))   
                              risk[r, c] = 0.5         

            for g in range(1, len(self.geese)):
                  if len(self.geese[g]):
                        risk[row_col(self.geese[g][0], self.columns)] = 1

            def spread(ri, ci, rj, cj):
                  if risk[ri, ci] < risk[rj, cj]*0.5:
                        risk[ri, ci] = risk[rj, cj]*0.5
                        bfs.put((ri, ci))

            while not bfs.empty():
                  r, c = bfs.get()
                  spread((r-1+self.rows)%self.rows, c, r, c)
                  spread((r+1)%self.rows, c, r, c)
                  spread(r, (c-1+self.columns)%self.columns, r, c)
                  spread(r, (c+1)%self.columns, r, c)
            
            return risk

      def reward_map(self):
            bfs = Queue()
            reward = np.zeros((self.rows, self.columns))

            for (r, c) in [row_col(f, self.columns) for f in self.foods]:
                  bfs.put((r, c))   
                  reward[r, c] = 1         

            def spread(ri, ci, rj, cj):
                  if reward[ri, ci] < reward[rj, cj]*0.9:
                        reward[ri, ci] = reward[rj, cj]*0.9
                        bfs.put((ri, ci))

            while not bfs.empty():
                  r, c = bfs.get()
                  spread((r-1+self.rows)%self.rows, c, r, c)
                  spread((r+1)%self.rows, c, r, c)
                  spread(r, (c-1+self.columns)%self.columns, r, c)
                  spread(r, (c+1)%self.columns, r, c)
            
            return reward
                  
cache = {}

def random_agent(obs, config):
      if not obs['index'] in cache:
            cache[obs['index']] = RandomAgent(config)
            
      return cache[obs['index']](obs)