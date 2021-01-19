import numpy as np
import random
import math
import settings
from operator import itemgetter


class dataset:
    def __init__(self, data):
        self.data = data
        self.stats = []
        # stats are given in order (XSD, YSD, XMEAN, YMEAN, CORR)

    def fitness(self, i):
        point = self.data[i]
        CP = (54.26, 47.83)
        # calculate distance from current point and closest point
        # and add to total
        dist = math.sqrt((CP[1] - point[1])**2 + (CP[0]-point[0])**2)
        return abs(dist-30)

    # returns position of closest point in dataset
    @staticmethod
    def closestPoint(data, point):
        array_data = np.asarray(data)
        dist = np.sum((array_data - point)**2, axis=1)
        return np.argmin(dist)

    # MovePointRandomly (dataset)
    def MovePointRandomly(self, i):
        k = 0.2
        # iterate through every x and y value
        magnitude_x = np.random.randn() * k
        magnitude_y = np.random.randn() * k
        self.data[i] = (self.data[i][0] + magnitude_x,
                        self.data[i][1] + magnitude_y)

    # getSummaryStats(dataset)
    def getStats(self):
        stats = []
        # calculate standard deviation of x and y and append
        stats.extend(np.std(self.data, axis=0))
        # calculate mean of x and y and add to stats
        stats.extend(np.mean(self.data, axis=0))
        # calculate correlation
        x, y = map(list, zip(*self.data))
        stats.append(np.corrcoef(x, y)[0, 1])
        return stats

    def statsClose(self, ds2):
        for i in range(len(self.stats)):
            # if stats arent same within 2 decimals
            if not (math.isclose(self.stats[i], ds2.stats[i], rel_tol=1e-02)):
                return False
        return True

    def generateData(self, i=None):
        data = []
        for _ in range(i):
            random_x = random.uniform(0, 100)
            random_y = random.uniform(0, 100)
            point = (random_x, random_y)
            data.append(point)
        return data

    def getBounds(self):
        bounds = []
        # min x, max x, min y, max y
        bounds.append(min(self.data)[0])
        bounds.append(max(self.data)[0])
        bounds.append(min(self.data, key=itemgetter(1))[1])
        bounds.append(max(self.data, key=itemgetter(1))[1])
        return bounds
