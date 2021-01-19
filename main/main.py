# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import utils
import settings
import copy
import math

# get an initial dataset
current_ds = utils.dataset(settings.initial_data)
current_ds.stats = current_ds.getStats()

# initializing graph for animation
plt.ion()
fig, ax = plt.subplots()
x, y = [], []
scat = ax.scatter(x, y, alpha=0.5)
plt.xlim(0, 100)
plt.ylim(0, 100)

for i in range(settings.generations):
    print("GEN: ", i)

    # decrease temperature
    a = i/settings.generations
    curve = 0.95**(100*a)
    current_temp = settings.initial_temp*curve

    # copy current to test
    test_ds = copy.copy(current_ds)
    # select a random member from neighbourhood
    position = np.random.randint(0, len(current_ds.data))
    # get cost before
    original_distance = current_ds.fitness(position)
    # move points
    test_ds.MovePointRandomly(position)
    # get fitness after
    current_distance = test_ds.fitness(position)
    # higher is better
    delta_e = original_distance - current_distance
    probability = math.exp(delta_e / current_temp)
    if (delta_e > 0):
        probability = 1
    # accept new solution if better OR if temperature probability passes
    if (current_distance < 1 or probability > np.random.uniform(0, 1) and
       current_ds.statsClose(test_ds)):
        # print("Successful!")
        current_ds = copy.copy(test_ds)
        if (i % 2000 == 0):
            x, y = map(list, zip(*current_ds.data))
            scat.set_offsets(np.c_[x, y])
            fig.canvas.draw_idle()
            plt.pause(0.1)

plt.waitforbuttonpress()
