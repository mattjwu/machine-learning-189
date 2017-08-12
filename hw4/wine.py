import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import csv
import sys

traindatafilename = "data"
data = scipy.io.loadmat(traindatafilename)

#Load and shuffle data sets
traindata = data['X']
NUM_FEATURES = traindata.shape[1]
NUM_SAMPLES = traindata.shape[0]
testdata = data['X_test']
labels = data['y']

temp = np.hstack((traindata, labels))
np.random.shuffle(temp)
X = temp[:,:NUM_FEATURES]

#Add bias term and normalize across all samples
tempmerge = np.vstack((X, testdata))
tempmerge = np.hstack((tempmerge, np.ones(6497).reshape(6497, 1)))
tempmerge = tempmerge / np.linalg.norm(tempmerge, axis=0)[np.newaxis, :]
X = tempmerge[:NUM_SAMPLES]
testdata = tempmerge[NUM_SAMPLES:]

y = temp[:,NUM_FEATURES:]

NUM_FEATURES += 1

TRAINING_SIZE = 5000
VALIDATION_SIZE = NUM_SAMPLES - TRAINING_SIZE

X_train = X[:TRAINING_SIZE]
y_train = y[:TRAINING_SIZE]

X_validation = X[TRAINING_SIZE:]
y_validation = y[TRAINING_SIZE:]

LAMBDA = .0001

def sigmoid(Xi, w):
    return 1 / (1 + np.exp(-np.dot(Xi.transpose(), w)))

def cost(XX, yy, w):
    norm_cost = LAMBDA * np.dot(w.transpose(), w)
    samples_cost = 0
    for i in range(XX.shape[0]):
        Xi, y_i = XX[i].transpose(), yy[i]
        if y_i == 1:
            samples_cost += np.log(sigmoid(Xi, w))
        else:
            samples_cost += np.log(1-sigmoid(Xi, w))
    return float(norm_cost - samples_cost)

def classify(XX, w):
    """Returns list of classified samples based on w"""
    s = [sigmoid(XX[i], w) for i in range(XX.shape[0])]
    p = [int(np.round(i)) for i in s]
    return p

def plot_batch():
    EPSILON = .1
    training_cost_list = []
    validation_cost_list = []
    num_iteration = []
    w = X_train[:1].transpose() #Initialize w
    training_cost_list.append(cost(X_train, y_train, w))
    validation_cost_list.append(cost(X_validation, y_validation, w))
    num_iteration.append(0)
    MAX_ITERATIONS = 3000
    for iteration in range(1, MAX_ITERATIONS + 1):
        s_vector = []
        for i in range(TRAINING_SIZE):
            Xi = X_train[i].transpose()
            s_vector.append(sigmoid(Xi, w))
        s_vector = np.array(s_vector).reshape(TRAINING_SIZE, 1)
        w = w + EPSILON * (np.dot(X_train.transpose(), y_train - s_vector)
            - 2 * LAMBDA * w)
        if iteration % 1 == 0:
            num_iteration.append(iteration)
            training_cost_list.append(cost(X_train, y_train, w))
            validation_cost_list.append(cost(X_validation, y_validation, w))
    print("Final training cost: " + str(training_cost_list[-1]))
    print("Final validation cost: " + str(validation_cost_list[-1]))

    val_guesses = classify(X_validation, w)
    total = 0
    wrong = 0
    for i in range(len(val_guesses)):
        if val_guesses[i] != y_validation[i]:
            wrong += 1
        total += 1
    print("Error Rate: " + str(wrong/total))
    sys.stdout.flush()

    plt.figure()
    plt.plot(num_iteration, training_cost_list)
    plt.title("Batch Descent, lambda="
        + str(LAMBDA) + ", epsilon=" + str(EPSILON))
    plt.ylabel("Cost")
    plt.xlabel("Number of Iterations")
    plt.figure()
    plt.plot(num_iteration, validation_cost_list)
    plt.title("Batch Descent Validation, lambda="
        + str(LAMBDA) + ", epsilon=" + str(EPSILON))
    plt.ylabel("Cost")
    plt.xlabel("Number of Iterations")
    plt.show()

def plot_stochastic(use_varying_epsilon=False):
    EPSILON = .1
    MULTIPLIER = 10000
    epsilon = EPSILON
    training_cost_list = []
    num_iteration = []
    w = X_train[:1].transpose() #Initialize w
    training_cost_list.append(cost(X_train, y_train, w))
    num_iteration.append(0)
    MAX_ITERATIONS = 20*TRAINING_SIZE
    for iteration in range(1, MAX_ITERATIONS + 1):
        i = iteration % TRAINING_SIZE
        Xi = X_train[i].reshape(NUM_FEATURES, 1)
        c = y_train[i] - sigmoid(Xi, w)
        if use_varying_epsilon:
            epsilon = EPSILON * MULTIPLIER / iteration
        w = w + epsilon * (c * Xi - 2 * LAMBDA * w)
        if iteration < 100 or iteration % 5 == 0 and iteration < 1000\
            or iteration % 50 == 0:
            num_iteration.append(iteration)
            training_cost_list.append(cost(X_train, y_train, w))
    print("Final training cost: " + str(training_cost_list[-1]))
    print("Final validation cost: " + str(cost(X_validation, y_validation, w)))
    val_guesses = classify(X_validation, w)
    total = 0
    wrong = 0
    for i in range(len(val_guesses)):
        if val_guesses[i] != y_validation[i]:
            wrong += 1
        total += 1
    print("Error Rate: " + str(wrong/total))
    sys.stdout.flush()

    if use_varying_epsilon:
        epsilon = EPSILON * MULTIPLIER

    plt.figure()
    plt.plot(num_iteration, training_cost_list)

    string = "Stochastic Descent, lambda="
    string += str(LAMBDA) + ", epsilon=" + str(epsilon)
    if use_varying_epsilon:
        string += "/iterations"
    plt.title(string)
    plt.ylabel("Cost")
    plt.xlabel("Number of Iterations")
    plt.show()

def classify_batch():
    global testdata
    EPSILON = .1
    num_iteration = []
    w = X[:1].transpose() #Initialize w
    num_iteration.append(0)
    MAX_ITERATIONS = 4000
    for iteration in range(1, MAX_ITERATIONS + 1):
        s_vector = []
        for i in range(TRAINING_SIZE):
            Xi = X[i].transpose()
            s_vector.append(sigmoid(Xi, w))
        s_vector = np.array(s_vector).reshape(TRAINING_SIZE, 1)
        eps = (2 - iteration / MAX_ITERATIONS / 2) * EPSILON
        w = w + eps * (np.dot(X_train.transpose(), y_train - s_vector)
            - 2 * LAMBDA * w)
        if iteration % 1 == 0:
            num_iteration.append(iteration)
    testdata = testdata / np.linalg.norm(testdata, axis=1)[:, np.newaxis]
    guesses = classify(testdata, w)
    with open('submission_1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        i = 0
        for g in guesses:
            writer.writerow([i, g])
            i += 1

#plot_batch()
#plot_stochastic(use_varying_epsilon=False)
#plot_stochastic(use_varying_epsilon=True)
#classify_batch()