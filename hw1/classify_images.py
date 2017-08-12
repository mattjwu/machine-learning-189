import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import csv

from sklearn.svm import SVC
from sklearn import metrics

traindatafilename = "hw01_data/cifar/train"
traindata = scipy.io.loadmat(traindatafilename)
traindata = traindata['trainX']

NUM_FEATURES = traindata.shape[1] - 1
VAL_SIZE = 5000

np.random.shuffle(traindata)

validation = traindata[:VAL_SIZE]
training = traindata[VAL_SIZE:]

validation_data = validation[:,:-1]
temp = validation[:,NUM_FEATURES:]
validation_labels = temp.reshape(temp.shape[0])

def generate_plot():
    training_examples_list = [100, 200, 500, 1000, 2000, 5000]
    accuracy_list = []
    for num_training_examples in training_examples_list:
        training_set = training[:num_training_examples]
        x = training_set[:,:-1]
        y = training_set[:,NUM_FEATURES:]
        y = y.reshape(y.shape[0])
        classifier = SVC(kernel='linear')
        classifier.fit(x, y)
        guesses = classifier.predict(validation_data)
        score = metrics.accuracy_score(validation_labels, guesses)
        accuracy_list.append(score)

    plt.plot(training_examples_list, accuracy_list, 'bo')
    plt.axis([0, 6000, .1, .4])

    plt.ylabel("Fraction Correct")
    plt.xlabel("Number of Training Images")

    for i in range(6):
        text = str(accuracy_list[i])
        x = training_examples_list[i]
        y = accuracy_list[i]
        plt.annotate(text, xy = (x, y), xytext = (x + 200, y - .01), fontsize = 10)

    plt.show()

def predict_test_set():
    x = traindata[:,:-1]
    y = traindata[:,NUM_FEATURES:]
    y = y.reshape(y.shape[0])
    classifier = SVC(kernel='linear')
    classifier.fit(x, y)
    testdatafilename = "hw01_data/cifar/test"
    testdata = scipy.io.loadmat(testdatafilename)
    testdata = testdata['testX']
    guesses = classifier.predict(testdata)
    with open('cifar.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        i = 0
        for g in guesses:
            writer.writerow([i, g])
            i += 1

generate_plot()
#predict_test_set()
