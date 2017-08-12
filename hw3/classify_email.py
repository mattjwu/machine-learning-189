import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import csv

traindatafilename = "dist/spam_data"
data = scipy.io.loadmat(traindatafilename)

traindata = data['training_data']
NUM_FEATURES = traindata.shape[1]
testdata = data['test_data']
labels = data['training_labels']
labels = labels.transpose()
traindata = np.append(traindata, labels, axis=1)
np.random.shuffle(traindata)

SIZE = traindata.shape[0]

def average_covariance_matrix(samples, labels):
    covs = []
    for i in range(2):
        sig_i = covariance_matrix_for_digit(i, samples, labels)
        covs.append(sig_i)
    a = np.mean(covs, axis=0)
    return np.mean(covs, axis=0)

def covariance_matrix_for_digit(digit, samples, labels):
    digit_samples = []
    for i in range(len(labels)):
        if labels[i] == digit:
            sample = samples[i]
            digit_samples.append(sample)
    sigma = np.cov(np.transpose(digit_samples))
    return sigma

def mean_for_digit(digit, samples, labels):
    digit_samples = []
    for i in range(len(labels)):
        if labels[i] == digit:
            sample = samples[i]
            digit_samples.append(sample)
    return np.mean(np.transpose(digit_samples), axis=1)

def predict_test_set():
    all_samples = traindata[:,:-1]
    temp = traindata[:,-1]
    all_labels = temp.reshape(temp.shape[0])
    covariance = average_covariance_matrix(all_samples, all_labels)
    if np.linalg.det(covariance) == 0:
        covariance += (10**-10)*np.eye(covariance.shape[0])
    precision = np.linalg.inv(covariance)
    mu_c = [mean_for_digit(i, all_samples, all_labels) for i in range(2)]
    mu_dot_precision = [np.dot(np.transpose(mu_c[i]), precision) for i in range(2)]
    guesses = []
    testdata = data['test_data']
    for sample in testdata:
        max_val = float('-inf')
        classification = -1
        for c in range(2):
            val = np.dot(mu_dot_precision[c], sample)
            val -= .5*np.dot(mu_dot_precision[c], mu_c[c])
            if val > max_val:
                max_val = val
                classification = c
        guesses.append(classification)

    with open('email_submission_2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        i = 0
        for g in guesses:
            writer.writerow([i, g])
            i += 1

predict_test_set()