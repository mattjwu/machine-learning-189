import numpy as np
from collections import Counter
import math

class DecisionTree(object):
    def __init__(self, categorical_vars = set(), max_depth = float('inf'), m = None):
        self.categorical_vars = categorical_vars
        self.max_depth = max_depth
        self.m = m

    class Node(object):
        def __init__(self, myTree, points, depth = 0):
            self.myTree = myTree
            self.is_leaf = False
            self.points = points
            self.depth = depth
            if self.should_stop():
                self.stop()
                return
            best_feature, best_split = myTree.segmenter(points)
            if best_feature == None:
                self.stop()
                return
            self.split_rule = (best_feature, best_split)
            left_points = []
            right_points = []
            if best_feature in myTree.categorical_vars:
                for p in points:
                    if self.myTree.data[p][best_feature] == best_split:
                        left_points.append(p)
                    else:
                        right_points.append(p)
            else:
                for p in points:
                    if self.myTree.data[p][best_feature] < best_split:
                        left_points.append(p)
                    else:
                        right_points.append(p)
            self.left = myTree.Node(self.myTree, left_points, depth + 1)
            self.right = myTree.Node(self.myTree, right_points, depth + 1)

        def should_stop(self):
            if self.depth >= self.myTree.max_depth:
                return True
            label = None
            for p in self.points:
                if not label:
                    label = self.myTree.labels[p]
                    continue
                if self.myTree.labels[p] != label:
                    return False
            return True

        def stop(self):
            frequencies = {}
            for p in self.points:
                l = float(self.myTree.labels[p])
                if l in frequencies:
                    frequencies[l] += 1
                else:
                    frequencies[l] = 1
            self.label = max(frequencies, key=frequencies.get)
            self.is_leaf = True

        def classify(self, point):
            if self.is_leaf:
                return self.label
            feature, split = self.split_rule
            if feature in self.myTree.categorical_vars:
                if point[feature] == split:
                    return self.left.classify(point)
                return self.right.classify(point)
            else:
                if point[feature] < split:
                    return self.left.classify(point)
                return self.right.classify(point)

        def get_path(self, point):
            if self.is_leaf:
                return [(self.label,)]
            feature, split = self.split_rule
            if feature in self.myTree.categorical_vars:
                if point[feature] == split:
                    return [(feature, split, "is")] + self.left.get_path(point)
                return [(feature, split, "isn't")] + self.right.get_path(point)
            else:
                if point[feature] < split:
                    return [(feature, split, "<")] + self.left.get_path(point)
                return [(feature, split, ">=")] + self.right.get_path(point)

        def decision_list(self):
            if self.is_leaf:
                return self.label
            left_lst = self.left.decision_list()
            right_lst = self.right.decision_list()
            return [self.split_rule, [left_lst, right_lst]]

    def train(self, data, labels, points=None):
        self.data = data
        self.labels = labels
        self.num_features = data.shape[1]
        if self.m == None:
            self.m = math.ceil(self.num_features**.5)
        if not points:
            points = [i for i in range(data.shape[0])]
        self.root = self.Node(self, points)

    def segmenter(self, points):
        total_labels = 0
        zero_labels = 0
        for p in points:
            if self.labels[p] == 0:
                zero_labels += 1
            total_labels += 1

        def find_split_for_feature(feature):
            best_entropy = float('inf')
            best_split = None
            if feature not in self.categorical_vars:
                left_zero_labels = 0
                left_total_labels = 0
                sorted_points = sorted(points, key=lambda p: self.data[p][feature])
                previous_value = self.data[sorted_points[0]][feature]
                if self.labels[sorted_points[0]] == 0:
                    left_zero_labels += 1
                left_total_labels += 1
                for i in range(1, len(sorted_points)):
                    new_value = self.data[sorted_points[i]][feature]
                    if previous_value != new_value:
                        new_entropy = self.impurity(
                            left_zero_labels,
                            left_total_labels - left_zero_labels,
                            zero_labels - left_zero_labels,
                            total_labels - left_total_labels - zero_labels + left_zero_labels)
                        if new_entropy < best_entropy:
                            best_entropy = new_entropy
                            best_split = (previous_value + new_value) / 2
                    previous_value = new_value
                    if self.labels[sorted_points[i]] == 0:
                        left_zero_labels += 1
                    left_total_labels += 1
            else:
                cat_total_freq = Counter()
                cat_zero_freq = Counter()
                for p in points:
                    category = self.data[p][feature]
                    label = self.labels[p]
                    cat_total_freq[category] += 1
                    if label == 0:
                        cat_zero_freq[category] += 1
                for category in cat_total_freq:
                    left_total_labels = cat_total_freq[category]
                    left_zero_labels = cat_zero_freq[category]
                    new_entropy = self.impurity(
                        left_zero_labels,
                        left_total_labels - left_zero_labels,
                        zero_labels - left_zero_labels,
                        total_labels - left_total_labels - zero_labels + left_zero_labels)
                    if new_entropy < best_entropy:
                        best_entropy = new_entropy
                        best_split = category
            return best_entropy, best_split

        #Pick random features to split on
        possible_split_features = self.select_random_features()
        best_entropy = float('inf')
        best_split = None
        best_feature = None
        for feature in possible_split_features:
            new_entropy, new_split = find_split_for_feature(feature)
            if new_entropy < best_entropy:
                best_entropy = new_entropy
                best_split = new_split
                best_feature = feature
        return best_feature, best_split

    def impurity(self, left_zeros, left_ones, right_zeros, right_ones):
        def H(prob):
            if prob == 0 or prob == 1:
                return 0
            return - prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)
        left_total = left_zeros + left_ones
        right_total = right_zeros + right_ones
        if left_total == 0 or right_total == 0:
            return float('inf')
        left_prob_zero = left_zeros / left_total
        right_prob_zero = right_zeros / right_total
        left_weight = left_total / (left_total + right_total)
        right_weight = 1 - left_weight
        return H(left_prob_zero) * left_weight + H(right_prob_zero) * right_weight

    def select_random_features(self):
        lst = [i for i in range(self.num_features)]
        np.random.shuffle(lst)
        return lst[:self.m]

    def predict(self, data):
        return self.root.classify(data)

    def get_first_split(self):
        return self.root.split_rule

    def get_path(self, data):
        return self.root.get_path(data)

    def get_decision_list(self):
        return self.root.decision_list()
