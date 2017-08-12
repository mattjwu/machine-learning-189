import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import csv
import math
import sys
from DecisionTree import DecisionTree
from RandomForest import RandomForest

#traindatafilename = "hw5_spam_dist/dist/spam_data"
traindatafilename = 'hw5_spam_dist/dist/default_spam_data'

data = scipy.io.loadmat(traindatafilename)

traindata = data['training_data']
NUM_FEATURES = traindata.shape[1]
testdata = data['test_data']
labels = data['training_labels']
labels = labels.transpose()
traindata = np.append(traindata, labels, axis=1)
np.random.shuffle(traindata)

SIZE = traindata.shape[0]

TRAINING_FRACTION = .9

N = math.ceil(SIZE * TRAINING_FRACTION)

traindata, labels = traindata[:,:-1], traindata[:, -1:]

training_data = traindata[:N]
training_labels = labels[:N]
validation_data = traindata[N:]
validation_labels = labels[N:]

num_training_points = N
num_validation_points = validation_data.shape[0]

def classify_with_decision_tree():
    tree = DecisionTree(max_depth = 25)
    tree.train(training_data, training_labels)

    num_right = 0
    for i in range(num_training_points):
        prediction = tree.predict(training_data[i])
        if prediction == training_labels[i]:
            num_right += 1
    print("Training Accuracy: " + str(num_right / num_training_points))

    num_right = 0
    for i in range(num_validation_points):
        prediction = tree.predict(validation_data[i])
        if prediction == validation_labels[i]:
            num_right += 1
    print("Validation Accuracy: " + str(num_right / num_validation_points))

def classify_with_random_forest():
    forest = RandomForest(num_trees = 25, max_depth = 25)
    forest.train(training_data, training_labels)

    num_right = 0
    for i in range(num_training_points):
        prediction = forest.predict(training_data[i])
        if prediction == training_labels[i]:
            num_right += 1
    print("Training Accuracy: " + str(num_right / num_training_points))

    num_right = 0
    for i in range(num_validation_points):
        prediction = forest.predict(validation_data[i])
        if prediction == validation_labels[i]:
            num_right += 1
    print("Validation Accuracy: " + str(num_right / num_validation_points))

def predict_test_data():
    forest = RandomForest(num_trees = 25, max_depth = 25)
    forest.train(training_data, training_labels)

    num_right = 0
    for i in range(num_training_points):
        prediction = forest.predict(training_data[i])
        if prediction == training_labels[i]:
            num_right += 1
    print("Training Accuracy: " + str(num_right / num_training_points))

    num_right = 0
    for i in range(num_validation_points):
        prediction = forest.predict(validation_data[i])
        if prediction == validation_labels[i]:
            num_right += 1
    print("Validation Accuracy: " + str(num_right / num_validation_points))

    guesses = []
    for i in range(testdata.shape[0]):
        point = testdata[i]
        guess = forest.predict(point)
        guesses.append(int(guess))

    with open('spam_1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        i = 0
        for g in guesses:
            writer.writerow([i, g])
            i += 1

feature_names = ['pain', 'private', 'bank', 'money', 'drug',
    'spam', 'prescription', 'creative', 'height', 'featured',
    'differ', 'width', 'other', 'energy', 'business', 'message',
    'volumes', 'revision', 'path', 'meter', 'memo', 'planning',
    'pleased', 'record', 'out', ';', '$', '#', '!', '(', '[', '&']

def get_path():
    tree = DecisionTree(max_depth = 10)
    tree.train(training_data, training_labels)
    def get_first_index(label):
        for i in range(num_training_points):
            if training_labels[i] == label:
                return i
    spam_point = training_data[get_first_index(0)]
    ham_point = training_data[get_first_index(1)]
    spam_path = tree.get_path(spam_point)
    ham_path = tree.get_path(ham_point)
    for decision in spam_path + ham_path:
        if len(decision) == 1:
            word = 'ham'
            if decision[0] == 1:
                word = 'spam'
            print("Therefore this email is " + word)
            print()
            continue
        feature, value, split_direction = decision
        name = feature_names[feature]
        print(name + ' '  + split_direction + ' ' + str(value))

def get_frequent_splits():
    forest = RandomForest(num_trees = 100, max_depth = 2)
    forest.train(training_data, training_labels)
    lst = forest.most_frequent_first_splits()
    for item in lst:
        word = ' < '
        split, frequency = item
        feature, value = split
        name = feature_names[feature]
        print(name + word + str(value) + ' (' + str(frequency) + ' trees)')

#classify_with_decision_tree()
#classify_with_random_forest()
#predict_test_data()
#get_path()
#get_frequent_splits()
