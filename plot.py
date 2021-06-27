import numpy as np
import os
from matplotlib import pyplot as plt

def plot(window=10, save=True):
      plt.figure(figsize=(10, 5))
      path = './logs/'
      def plotting(log, subplt, window=10):
            plt.subplot(1, 2, subplt)
            plot = [np.mean(log[i-window:i]) for i in range(window, len(log)+1, window)]

            plt.scatter(2, np.max(log), marker="$max$", color='black', s=1000)
            plt.scatter(2, np.min(log), marker="$min$", color='black', s=1000)
            plt.plot(np.arange(len(plot)), plot)

      plotting(np.loadtxt(path+'log.txt'), 1)
      plotting(np.loadtxt(path+'surviving.txt'), 2)
      if save:
            plt.savefig(path+'plot.png')
      else:
            plt.show()

if __name__ == '__main__':
      plot(window=5, save=False)