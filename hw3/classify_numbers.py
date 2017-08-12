import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import csv
import sys

traindatafilename = "hw3_mnist_dist/train"
traindata = scipy.io.loadmat(traindatafilename)
traindata = traindata['trainX']

VAL_SIZE = 10000

def normalize(sample):
    return sample / np.linalg.norm(sample)

def visualize_covariance_matrix(digit):
    digit_data = []
    for sample in traindata:
        if sample[-1] == digit:
            digit_data.append(normalize(sample[:-1]))
    sigma = np.cov(np.transpose(digit_data))
    plt.imshow(sigma, interpolation='nearest')
    #plt.colorbar()
    plt.show()

def average_covariance_matrix(samples, labels):
    covs = []
    for i in range(10):
        sig_i = covariance_matrix_for_digit(i, samples, labels)
        covs.append(sig_i)
    a = np.mean(covs, axis=0)
    return np.mean(covs, axis=0)

def covariance_matrix_for_digit(digit, samples, labels):
    digit_samples = []
    for i in range(len(labels)):
        if labels[i] == digit:
            sample = samples[i]
            digit_samples.append(normalize(sample))
    sigma = np.cov(np.transpose(digit_samples))
    return sigma

def mean_for_digit(digit, samples, labels):
    digit_samples = []
    for i in range(len(labels)):
        if labels[i] == digit:
            sample = samples[i]
            digit_samples.append(normalize(sample))
    return np.mean(np.transpose(digit_samples), axis=1)

np.random.shuffle(traindata)
validation = traindata[:VAL_SIZE]
training = traindata[VAL_SIZE:]

training_samples = training[:,:-1]
temp = training[:,784:]
training_labels = temp.reshape(temp.shape[0])

validation_samples = validation[:,:-1]
temp = validation[:,784:]
validation_labels = temp.reshape(temp.shape[0])

def plot_lda():
    num_points = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
    accuracy_list = []
    for N in num_points:
        print(N)
        sys.stdout.flush()
        temp_samples = training_samples[:N]
        temp_labels = training_labels[:N]
        covariance = average_covariance_matrix(temp_samples, temp_labels)
        if np.linalg.det(covariance) == 0:
            covariance += (10**-10)*np.eye(784)
        precision = np.linalg.inv(covariance)
        mu_c = [mean_for_digit(i, temp_samples, temp_labels) for i in range(10)]
        mu_dot_precision = [np.dot(np.transpose(mu_c[i]), precision) for i in range(10)]
        wrong = 0
        for i in range(VAL_SIZE):
            sample = validation_samples[i]
            sample = normalize(sample)
            max_val = float('-inf')
            classification = -1
            for c in range(10):
                val = np.dot(mu_dot_precision[c], sample)
                val -= .5*np.dot(mu_dot_precision[c], mu_c[c])
                if val > max_val:
                    max_val = val
                    classification = c
            if classification != validation_labels[i]:
                wrong += 1
        accuracy_list.append(wrong / VAL_SIZE)
    plt.plot(num_points, accuracy_list, 'bo')
    plt.axis([0, 55000, 0, 1])
    plt.title("LDA Error Rate")
    plt.ylabel("Error Rate")
    plt.xlabel("Number of Training Samples")
    for i in range(len(accuracy_list)):
        text = str(accuracy_list[i])
        x = num_points[i]
        y = accuracy_list[i]
        plt.annotate(text, xy = (x, y), xytext = (x + 560, y - .03), fontsize = 10)
    print(accuracy_list)
    sys.stdout.flush()
    plt.show()

def plot_qda():
    num_points = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
    accuracy_list = []
    for N in num_points:
        print(N)
        sys.stdout.flush()
        temp_samples = training_samples[:N]
        temp_labels = training_labels[:N]
        covariance_c = []
        for i in range(10):
            cov = covariance_matrix_for_digit(i, temp_samples, temp_labels)
            if np.linalg.det(cov) == 0:
                cov += (1)*np.eye(784)
            covariance_c.append(cov)
        precision_c = [np.linalg.inv(cov) for cov in covariance_c]
        log_det_c = [np.log(np.linalg.det(cov)) for cov in covariance_c]
        mu_c = [mean_for_digit(i, temp_samples, temp_labels) for i in range(10)]
        #mu_dot_precision = [np.dot(np.transpose(mu_c[i]), precision_c[i]) for i in range(10)]
        wrong = 0
        for i in range(VAL_SIZE):
            sample = validation_samples[i]
            sample = normalize(sample)
            max_val = float('-inf')
            classification = -1
            for c in range(10):
                y = sample - mu_c[c]
                val = -np.dot(np.dot(np.transpose(y), precision_c[c]), y)
                val -= log_det_c[c]
                if val > max_val:
                    max_val = val
                    classification = c
            if classification != validation_labels[i]:
                wrong += 1
        accuracy_list.append(wrong / VAL_SIZE)
    plt.plot(num_points, accuracy_list, 'bo')
    plt.axis([0, 55000, 0, 1])
    plt.title("QDA Error Rate")
    plt.ylabel("Error Rate")
    plt.xlabel("Number of Training Samples")
    for i in range(len(accuracy_list)):
        text = str(accuracy_list[i])
        x = num_points[i]
        y = accuracy_list[i]
        plt.annotate(text, xy = (x, y), xytext = (x + 560, y - .03), fontsize = 10)
    print(accuracy_list)
    sys.stdout.flush()
    plt.show()

def predict_test_set():
    all_samples = traindata[:,:-1]
    temp = traindata[:,784:]
    all_labels = temp.reshape(temp.shape[0])
    covariance = average_covariance_matrix(all_samples, all_labels)
    if np.linalg.det(covariance) == 0:
        covariance += (10**-10)*np.eye(784)
    precision = np.linalg.inv(covariance)
    mu_c = [mean_for_digit(i, all_samples, all_labels) for i in range(10)]
    mu_dot_precision = [np.dot(np.transpose(mu_c[i]), precision) for i in range(10)]
    guesses = []
    testdatafilename = "hw3_mnist_dist/test"
    testdata = scipy.io.loadmat(testdatafilename)
    testdata = testdata['testX']
    for sample in testdata:
        x = normalize(sample)
        max_val = float('-inf')
        classification = -1
        for c in range(10):
            val = np.dot(mu_dot_precision[c], x)
            val -= .5*np.dot(mu_dot_precision[c], mu_c[c])
            if val > max_val:
                max_val = val
                classification = c
        guesses.append(classification)

    with open('mnist_submission_1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        i = 0
        for g in guesses:
            writer.writerow([i, g])
            i += 1

#visualize_covariance_matrix(6)
plot_lda()
plot_qda()
#predict_test_set()
