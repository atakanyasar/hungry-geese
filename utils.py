import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, row_col
from queue import Queue

rows = 7
columns = 11

path = [
      [-1, 0],
      [0, 1],
      [1, 0],
      [0, -1]
]
num = {
      Action.NORTH:0, 
      Action.EAST:1, 
      Action.SOUTH:2, 
      Action.WEST:3,
      'NORTH':0,
      'EAST':1,
      'SOUTH':2,
      'WEST':3,
      None:None
}

def go(p, k):
      if k in list(num.keys()):
            k = num[k]
      return ((p[0] + path[k][0] + rows) % rows, (p[1] + path[k][1] + columns) % columns)

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


def BFS(s, Grid=None, let=[], v=0, from_head=False):
      grid = Grid.copy()

      if len(let) > 0:
            grid[grid > 0] = 100
      
      bfs = Queue()
      val = np.zeros((rows, columns), int)
      val.fill(100)

      if from_head or grid[s] <= v:
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

def reverseBFS(s, v, Grid=None):
      grid = Grid.copy()
      
      bfs = Queue()
      val = np.zeros((rows, columns), int)

      bfs.put(s)
      val[s] = v
      
      def spread(y, x):
            if val[y] < val[x] - 1 and grid[y]>=val[x]-1:
                  val[y] = val[x] - 1
                  bfs.put(y)
            
      while not bfs.empty():
            x = bfs.get()
            for k in range(4):
                  spread(go(x, k), x)
      return val

def pair(x):
      return row_col(x, columns)
