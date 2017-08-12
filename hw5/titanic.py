import numpy as np
import scipy.io
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
import sys
from DecisionTree import DecisionTree
from RandomForest import RandomForest
from collections import Counter
import math

TRAINING_FRACTION = .85

train_filename = 'hw5_titanic_dist/titanic_training.csv'
test_filename = 'hw5_titanic_dist/titanic_testing_data.csv'

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
    if data['survived'] == '':
        del training_data[i]
        continue
    del data['ticket']
    del data['cabin']
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
    del data['ticket']
    del data['cabin']
    for key in data:
        try:
            data[key] = float(data[key])
        except:
            if data[key] == '':
                data[key] = np.nan
            continue
    i += 1

labels = np.array([[sample['survived']] for sample in training_data])

all_data = training_data + testing_data

continuous_data = np.array([[sample['pclass'], sample['age'],
    sample['sibsp'], sample['parch'], sample['fare']] for sample in all_data])

V = Imputer()
continuous_data = V.fit_transform(continuous_data)

categorical_data = [[sample['sex'], sample['embarked']] for sample in all_data]

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

cat_set = set([5, 6])

def classify_with_decision_tree():
    tree = DecisionTree(max_depth = 7, categorical_vars = cat_set)
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
    forest = RandomForest(num_trees = 250, max_depth = 7, categorical_vars = cat_set)
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

feature_names = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex', 'embarked']

def output_tree():
    tree = DecisionTree(max_depth = 3, categorical_vars = cat_set)
    tree.train(training_data, training_labels)
    lst = tree.get_decision_list()

    def output_node(lst, depth=0):
        if type(lst) == list:
            feature, value = lst[0]
            left, right = ' < ', ' >= '
            if feature in cat_set:
                value = inverse_list[feature - CONTINUOUS_FEATURES][value]
                left, right = ' is ', " isn't "
            name = feature_names[feature]
            print('    ' * depth + name + left + str(value))
            output_node(lst[1][0], depth + 1)
            print('    ' * depth + name + right + str(value))
            output_node(lst[1][1], depth + 1)
        else:
            label = 'survived' if lst == 1 else 'died'
            print('    ' * depth + label)

    output_node(lst)

def predict_test_data():
    forest = RandomForest(num_trees = 250, max_depth = 7, categorical_vars = cat_set)
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
    for i in range(TEST_SIZE):
        point = testing_data[i]
        guess = tree.predict(point)
        guesses.append(int(guess))

    with open('titanic_1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        i = 1
        for g in guesses:
            writer.writerow([i, g])
            i += 1

#classify_with_decision_tree()
#classify_with_random_forest()
#output_tree()
#predict_test_data()
