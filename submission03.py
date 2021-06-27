import numpy as np
from queue import Queue
from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration, Action, Observation, row_col
import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, row_col
from queue import Queue

rows = 7
columns = 11

def make_grid(geese):
      grid = np.zeros((7, 11), 'int')
      
      for g in range(len(geese)):
            for rc in geese[g]:
                  grid[row_col(rc, columns)] = g+1
      return grid


def adjust(grid, geese, us, foods):

      for rc in geese[us]:
            grid[row_col(rc, columns)] = 1
      for rc in geese[0]:
            grid[row_col(rc, columns)] = us+1

      geese[0], geese[us] = geese[us], geese[0]

      while geese[0][0] // columns != 3:
            grid = shift_vertical(grid, geese, foods)
      
      while geese[0][0] % columns != 5:
            grid = shift_horizontal(grid, geese, foods)
      
      return grid


def shift_vertical(grid, geese, foods):
      grid = np.concatenate([grid[1:], grid[:1]], axis=0)
      for g in range(len(geese)):
            for j in range(len(geese[g])):
                  geese[g][j] = (geese[g][j] - columns + rows*columns) % (rows*columns)
      for f in range(len(foods)):
            foods[f] = (foods[f] - columns + rows*columns) % (rows*columns)
      return grid

def shift_horizontal(grid, geese, foods):
      grid = np.concatenate([grid[:,1:], grid[:,:1]], axis=1)

      for g in range(len(geese)):
            for j in range(len(geese[g])):
                  geese[g][j] = geese[g][j]-geese[g][j]%columns + (geese[g][j]%columns-1+columns) % columns
      
      for f in range(len(foods)):
            foods[f] = foods[f]-foods[f]%columns + (foods[f]%columns-1+columns) % columns
      
      return grid


def BFS(s, grid=None, let=[], v=0):

      if len(let) > 0:
            grid[grid > 0] = 100
      
      bfs = Queue()
      val = np.zeros((rows, columns), int)
      val.fill(100)

      if grid[s] <= v:
            bfs.put(s)
            val[s] = v
      
      def spread(y, x):
            if val[y] > val[x] + 1 and ((grid[y]<=val[x]+1) or (y in let)):
                  val[y] = val[x] + 1
                  if grid[y] <= val[y]:
                        bfs.put(y)
      while not bfs.empty():
            r, c = bfs.get()
            spread(((r-1+rows)%rows, c), (r, c))
            spread(((r+1)%rows, c), (r, c))
            spread((r, (c-1+columns)%columns), (r, c))
            spread((r, (c+1)%columns), (r, c))
      
      return val

def pair(x):
      return row_col(x, columns)

def will_grow(goose, foods):
      r, c = goose[0]
      if ((r-1+rows)%rows, c) in foods:
            return True
      if ((r+1)%rows, c) in foods:
            return True
      if (r, (c-1+columns)%columns) in foods:
            return True
      if (r, (c+1)%columns) in foods:
            return True
      return False
class SmartAgent:
      def __init__(self, config):
            self.max_eps = config['episodeSteps']
            self.time_out = config['actTimeout']
            self.columns = config['columns']
            self.rows = config['rows']
            self.hunger_rate = config['hunger_rate']

            self.move = None
            

      def __call__(self, obs):
            # print(obs)
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