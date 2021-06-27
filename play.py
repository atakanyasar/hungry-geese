import os
from IPython.display import display
from kaggle_environments import make, evaluate, envs
from greedy_agent import greedy_agent
from random_agent import random_agent
from src.reinforced_agent import reinforced_agent
from submission02 import smart_agent as submission02
from submission03 import smart_agent as submission03

def play(agents):
    env = make('hungry_geese', debug=True, configuration={'episodeSteps':200})
    env.run(agents)
    open("game.html", "w").write(env.render(mode="html", width=500, height=450))
    os.system("game.html")

if __name__ == '__main__':
    agents = [reinforced_agent, reinforced_agent, reinforced_agent, reinforced_agent]
    play(agents)