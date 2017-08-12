import numpy as np
import math
from collections import Counter
from DecisionTree import DecisionTree
import sys

class RandomForest(object):
    def __init__(self, num_trees, num_sample_points = None,
        categorical_vars=set(), max_depth=float('inf'), m = None):
        self.num_trees = num_trees
        self.num_sample_points = num_sample_points
        self.categorical_vars = categorical_vars
        self.max_depth = max_depth
        self.m = m

    def train(self, data, labels):
        self.data = data
        self.labels = labels
        self.num_features = data.shape[1]
        self.N = data.shape[0]
        if self.num_sample_points == None:
            self.num_sample_points = self.N
        if self.m == None:
            self.m = math.ceil(self.num_features**.5)
        self.trees = []
        for i in range(self.num_trees):
            points = self.draw_sample()
            new_tree = DecisionTree(self.categorical_vars, self.max_depth, self.m)
            new_tree.train(self.data, self.labels, points)
            self.trees.append(new_tree)
            #print("Finished training tree " + str(i + 1))
            #sys.stdout.flush()

    def predict(self, data):
        counts = Counter()
        for tree in self.trees:
            prediction = tree.predict(data)
            counts[prediction] += 1
        prediction = max(counts, key=counts.get)
        return prediction

    def most_frequent_first_splits(self):
        counts = Counter()
        for tree in self.trees:
            split = tree.get_first_split()
            counts[split] += 1
        lst = sorted(counts, key=counts.get)
        return [(i, counts[i]) for i in lst][::-1]

    def draw_sample(self):
        return [np.random.randint(self.N) for i in range(self.num_sample_points)]