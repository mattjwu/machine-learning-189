import numpy as np
import scipy.io
import csv
from sklearn.preprocessing import Imputer
import sys
from DecisionTree import DecisionTree
from RandomForest import RandomForest
from collections import Counter
import math
import matplotlib.pyplot as plt

TRAINING_FRACTION = .8

train_filename = 'hw5_census_dist/train_data.csv'
test_filename = 'hw5_census_dist/test_data.csv'

training_data = []
testing_data = []

with open(train_filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        training_data.append(row)

with open(test_filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        testing_data.append(row)

i = 0
while i < len(training_data):
    data = training_data[i]
    if data['label'] == '':
        del training_data[i]
        continue
    for key in data:
        try:
            data[key] = float(data[key])
        except:
            if data[key] == '':
                data[key] = np.nan
            continue
    i += 1

i = 0
while i < len(testing_data):
    data = testing_data[i]
    for key in data:
        try:
            data[key] = float(data[key])
        except:
            if data[key] == '':
                data[key] = np.nan
            continue
    i += 1

labels = np.array([[sample['label']] for sample in training_data])

all_data = training_data + testing_data

continuous_data = np.array([[sample['age'], sample['fnlwgt'],
    sample['education-num'], sample['capital-gain'],
    sample['capital-loss'], sample['hours-per-week']] for sample in all_data])

V = Imputer()
continuous_data = V.fit_transform(continuous_data)

categorical_data = [[sample['workclass'], sample['education'],
    sample['marital-status'], sample['occupation'], sample['relationship'],
    sample['race'], sample['sex'], sample['native-country']] for sample in all_data]

#Use the most frequent occurence to replace blanks in categorical data
num_cat_rows = len(categorical_data)
num_cat_feat = len(categorical_data[0])

for col in range(num_cat_feat):
    counts = Counter()
    impute = False
    for row in range(num_cat_rows):
        val = categorical_data[row][col]
        if val != np.nan:
            counts[val] += 1
        else:
            impute = True
    if impute:
        new_val = max(counts, key=counts.get)
        for row in range(num_cat_rows):
            if categorical_data[row][col] == np.nan:
                categorical_data[row][col] = new_val

#Convert categorical data into floats
translation_list = []
inverse_list = []
for col in range(num_cat_feat):
    translator = {}
    inverse = {}
    i = 1
    for row in range(num_cat_rows):
        val = categorical_data[row][col]
        if val in translator:
            categorical_data[row][col] = translator[val]
        else:
            translator[val] = float(i)
            categorical_data[row][col] = translator[val]
            inverse[i] = val
            i += 1
    translation_list.append(translator)
    inverse_list.append(inverse)

categorical_data = np.array(categorical_data)

TRAIN_SIZE = len(training_data)
TEST_SIZE = len(testing_data)
CONTINUOUS_FEATURES = continuous_data.shape[1]
CATEGORICAL_FEATURES = categorical_data.shape[1]
NUM_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

all_data = np.hstack((continuous_data, categorical_data))
traindata = all_data[:TRAIN_SIZE]
testing_data = all_data[TRAIN_SIZE:]

#Shuffling the data and setting aside a validation set
traindata = np.append(traindata, labels, axis=1)
np.random.shuffle(traindata)

SIZE = traindata.shape[0]

N = math.ceil(SIZE * TRAINING_FRACTION)

traindata, labels = traindata[:,:-1], traindata[:, -1:]

training_data = traindata[:N]
training_labels = labels[:N]
validation_data = traindata[N:]
validation_labels = labels[N:]

num_training_points = N
num_validation_points = validation_data.shape[0]
#Validation set end

cat_set = set([6, 7, 8, 9, 10, 11, 12, 13])

def classify_with_decision_tree():
    tree = DecisionTree(max_depth = 10, categorical_vars = cat_set)
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
    forest = RandomForest(num_trees = 25, max_depth = 10, categorical_vars = cat_set)
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

feature_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
        'hours-per-week', 'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country']

def get_path():
    tree = DecisionTree(max_depth = 10, categorical_vars = cat_set)
    tree.train(training_data, training_labels)
    def get_first_index(label):
        for i in range(num_training_points):
            if training_labels[i] == label:
                return i
    poor_point = training_data[get_first_index(0)]
    rich_point = training_data[get_first_index(1)]
    poor_path = tree.get_path(poor_point)
    rich_path = tree.get_path(rich_point)
    for decision in poor_path + rich_path:
        if len(decision) == 1:
            word = 'less'
            if decision[0] == 1:
                word = 'more'
            print("Therefore this person makes " + word + " than $50k")
            print()
            continue
        feature, value, split_direction = decision
        if feature in cat_set:
            value = inverse_list[feature - CONTINUOUS_FEATURES][value]
        name = feature_names[feature]
        print(name + ' '  + split_direction + ' ' + str(value))

def get_frequent_splits():
    forest = RandomForest(num_trees = 100, max_depth = 2, categorical_vars = cat_set)
    forest.train(training_data, training_labels)
    lst = forest.most_frequent_first_splits()
    for item in lst:
        word = ' < '
        split, frequency = item
        feature, value = split
        if feature in cat_set:
            value = inverse_list[feature - CONTINUOUS_FEATURES][value]
            word = ' is '
        name = feature_names[feature]
        print(name + word + str(value) + ' (' + str(frequency) + ' trees)')

def graph_accuracy():
    accuracy = []
    num_trees = []
    for j in range(5, 41, 5):
        forest = RandomForest(num_trees = j, max_depth = 10, categorical_vars = cat_set)
        forest.train(training_data, training_labels)
        num_right = 0
        for i in range(num_validation_points):
            prediction = forest.predict(validation_data[i])
            if prediction == validation_labels[i]:
                num_right += 1
        accuracy.append(num_right / num_validation_points)
        num_trees.append(j)
        print(j)
        sys.stdout.flush()
    plt.figure()
    plt.plot(num_trees, accuracy)
    plt.title("Census Accuracy For Random Forest")
    plt.ylabel("Accuracy Rate")
    plt.xlabel("Number of Trees")
    plt.show()

def predict_test_data():
    tree = DecisionTree(max_depth=10, categorical_vars = cat_set)
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

    guesses = []
    for i in range(TEST_SIZE):
        point = testing_data[i]
        guess = tree.predict(point)
        guesses.append(int(guess))

    with open('census_1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        i = 1
        for g in guesses:
            writer.writerow([i, g])
            i += 1


#classify_with_decision_tree()
#classify_with_random_forest()
#get_path()
#get_frequent_splits()
#predict_test_data()
#graph_accuracy()