import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import csv

from sklearn.svm import SVC
from sklearn import metrics

traindatafilename = "hw01_data/mnist/train"
traindata = scipy.io.loadmat(traindatafilename)
traindata = traindata['trainX']

VAL_SIZE = 10000

np.random.shuffle(traindata)

validation = traindata[:VAL_SIZE]
training = traindata[VAL_SIZE:]

validation_data = validation[:,:-1]
temp = validation[:,784:]
validation_labels = temp.reshape(temp.shape[0])

def generate_plot():
    training_examples_list = [100, 200, 500, 1000, 2000, 5000, 10000]
    accuracy_list = []
    for num_training_examples in training_examples_list:
        training_set = training[:num_training_examples]
        x = training_set[:,:-1]
        y = training_set[:,784:]
        y = y.reshape(y.shape[0])
        classifier = SVC(kernel='linear')
        classifier.fit(x, y)
        guesses = classifier.predict(validation_data)
        score = metrics.accuracy_score(validation_labels, guesses)
        accuracy_list.append(score)

    plt.plot(training_examples_list, accuracy_list, 'bo')
    plt.axis([0, 11000, .5, 1])
    plt.ylabel("Fraction Correct")
    plt.xlabel("Number of Training Numbers")

    for i in range(7):
        text = str(accuracy_list[i])
        x = training_examples_list[i]
        y = accuracy_list[i]
        plt.annotate(text, xy = (x, y), xytext = (x + 160, y - .015), fontsize = 10)

    plt.show()

def find_c_value():
    #c_values = [10**-x for x in range(11)]
    #Everything from 10**0 to 10**-5 makes no difference
    #C=10**-6 is just barely better than C=1
    #C=10**-7 is just barely worse than C=1
    #C=10**-8 is significantly worse than C=1

    #The optimal value for C is likely between 10**-5 and 10**-7

    c_values = [(10**-7)*(100**(1/10))**x for x in range(11)]

    power_list = []
    accuracy_list = []

    for i in range(11):
        c = c_values[i]
        #num_training_examples = 1000
        num_training_examples = 10000
        training_set = training[:num_training_examples]
        x = training_set[:,:-1]
        y = training_set[:,784:]
        y = y.reshape(y.shape[0])
        classifier = SVC(kernel='linear', C = c)
        classifier.fit(x, y)
        guesses = classifier.predict(validation_data)
        score = metrics.accuracy_score(validation_labels, guesses)
        power_list.append(i)
        accuracy_list.append(score)
    plt.plot(power_list, accuracy_list, 'bo')
    plt.axis([-1, 11, .9, .95])
    plt.ylabel("Fraction Correct")
    plt.xlabel("Value of C (C=(1e-7)*(100^(1/10))^x)")

    for i in range(len(power_list)):
        text = str(accuracy_list[i])[1:]
        x = power_list[i]
        y = accuracy_list[i]
        plt.annotate(text, xy = (x, y), xytext = (x + .15, y - .002), fontsize = 9)

    plt.show()

def predict_test_set():
    c = 1e-7*(100**(1/10))**6
    x = traindata[:,:-1]
    y = traindata[:,784:]
    y = y.reshape(y.shape[0])
    classifier = SVC(kernel='linear', C = c)
    classifier.fit(x, y)
    testdatafilename = "hw01_data/mnist/test"
    testdata = scipy.io.loadmat(testdatafilename)
    testdata = testdata['testX']
    guesses = classifier.predict(testdata)
    with open('mnist_3.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        i = 0
        for g in guesses:
            writer.writerow([i, g])
            i += 1

#generate_plot()
#find_c_value()
predict_test_set()
