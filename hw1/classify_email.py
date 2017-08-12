import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import csv

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.cross_validation import KFold

traindatafilename = "hw01_data/spam/spam_data"
data = scipy.io.loadmat(traindatafilename)

traindata = data['training_data']
NUM_FEATURES = traindata.shape[1]
testdata = data['test_data']
labels = data['training_labels']
labels = labels.transpose()
traindata = np.append(traindata, labels, axis=1)
np.random.shuffle(traindata)

SIZE = traindata.shape[0]

def generate_plot():
    VAL_FRACTION = .2
    validation = traindata[:int(SIZE * VAL_FRACTION)]
    training = traindata[int(SIZE * VAL_FRACTION):]
    MAX_TRAIN_NUM = training.shape[0]
    validation_data = validation[:,:-1]
    temp = validation[:,NUM_FEATURES:]
    validation_labels = temp.reshape(temp.shape[0])
    training_examples_list = [100, 200, 500, 1000, 2000, MAX_TRAIN_NUM]
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
    plt.axis([0, 5000, .7, 1])
    plt.ylabel("Fraction Correct")
    plt.xlabel("Number of Training Emails")

    for i in range(6):
        text = str(round(accuracy_list[i], 4))
        x = training_examples_list[i]
        y = accuracy_list[i]
        plt.annotate(text, xy = (x, y), xytext = (x + 80, y - .005), fontsize = 10)

    plt.show()

def find_c_value():
    c_values = [.1, .3, 1, 3, 10]

    power_list = []
    accuracy_list = []

    kf = KFold(SIZE, n_folds=5)

    for c in c_values:
        total_score = 0
        for train_index, test_index in kf:
            training_set, validation_set = traindata[train_index], traindata[test_index]

            x = training_set[:,:-1]
            y = training_set[:,NUM_FEATURES:]
            y = y.reshape(y.shape[0])
            classifier = SVC(kernel='linear', C = c)
            classifier.fit(x, y)

            validation_data = validation_set[:,:-1]
            validation_labels = validation_set[:,NUM_FEATURES:]
            guesses = classifier.predict(validation_data)
            total_score += metrics.accuracy_score(validation_labels, guesses)

        score = total_score / 5
        power_list.append(c)
        accuracy_list.append(score)
    plt.plot(power_list, accuracy_list, 'bo')
    plt.axis([-1, 12, .85, .95])
    plt.ylabel("Fraction Correct")
    plt.xlabel("Value of C")

    for i in range(len(power_list)):
        text = str(round(accuracy_list[i], 4))
        x = power_list[i]
        y = accuracy_list[i]
        plt.annotate(text, xy = (x, y), xytext = (x + .25, y - .002), fontsize = 10)

    plt.show()

def predict_test_set():
    c = 10
    x = traindata[:,:-1]
    y = traindata[:,NUM_FEATURES:]
    y = y.reshape(y.shape[0])
    classifier = SVC(kernel='linear', C = c)
    classifier.fit(x, y)
    guesses = classifier.predict(testdata)
    with open('email.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        i = 0
        for g in guesses:
            writer.writerow([i, int(g)])
            i += 1

#generate_plot()
find_c_value()
#predict_test_set()
