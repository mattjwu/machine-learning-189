import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import csv
import scipy.misc
import sys

traindatafilename = "hw6_data_dist/letters_data"
data = scipy.io.loadmat(traindatafilename)

TRAINING_FRACTION = .8

traindata = data['train_x']
trainlabels = data['train_y']

NUM_FEATURES = traindata.shape[1]
NUM_SAMPLES = traindata.shape[0]

testdata = data['test_x']

#Shuffle the training data
temp = np.hstack((traindata, trainlabels))
np.random.shuffle(temp)
X = temp[:,:NUM_FEATURES]
y = temp[:,NUM_FEATURES:]

#Add bias terms and normalize all data
"""
temp = np.vstack((X, testdata))
temp = np.hstack((temp, np.ones(temp.shape[0]).reshape(temp.shape[0], 1)))
temp = temp / (np.linalg.norm(temp, axis=0)[np.newaxis, :] + .000000001)
temp = temp - np.mean(temp, axis=0)
X = temp[:NUM_SAMPLES]
testdata = temp[NUM_SAMPLES:]
"""

column_means = np.mean(X, axis=0)

X = X - column_means
testdata = testdata - column_means

column_stds = np.std(X, axis=0) + .000000001
X = X / column_stds
testdata = testdata / column_stds

X = np.hstack((X, np.ones(X.shape[0]).reshape(X.shape[0], 1)))
testdata = np.hstack((testdata, np.ones(testdata.shape[0]).reshape(testdata.shape[0], 1)))

#Increment features by 1 to account for bias
NUM_FEATURES += 1

#Create training and validation sets
TRAINING_SIZE = int(TRAINING_FRACTION * NUM_SAMPLES)
VALIDATION_SIZE = NUM_SAMPLES - TRAINING_SIZE
sys.stdout.flush()

X_train = X[:TRAINING_SIZE]
y_train = y[:TRAINING_SIZE]

X_validation = X[TRAINING_SIZE:]
y_validation = y[TRAINING_SIZE:]

#END PREPROCESSING DATA

#Initialize constants and matrices
EPSILON = .002
decay_factor = .85
V = np.random.normal(0, .01, (200, 785))
W = np.random.normal(0, .01, (26, 201))

y_array = [None]
for i in range(26):
    temp = np.ones((26, 1)) * .1
    temp[i] = .9
    y_array.append(temp)

def sigmoid(num):
    return 1 / (1 + np.exp(-num))

def classify(sample):
    h = np.tanh(np.dot(V, sample)).reshape((200, 1))
    h1 = np.vstack((h, 1))
    z = sigmoid(np.dot(W, h1))
    return np.argmax(z) + 1

def caclulate_validation_accuracy():
    right = 0
    for i in range(VALIDATION_SIZE):
        sample = X_validation[i]
        guess = classify(sample)
        label = y_validation[i]
        if guess == label:
            right += 1
    return right / VALIDATION_SIZE

def calculate_training_accuracy():
    right = 0
    for i in range(TRAINING_SIZE):
        sample = X_train[i]
        guess = classify(sample)
        label = y_train[i]
        if guess == label:
            right += 1
    return right / TRAINING_SIZE

def calculate_training_loss():
    total_loss = 0
    for i in range(TRAINING_SIZE):
        x = X_train[i]
        h = np.tanh(np.dot(V, x).reshape((200, 1)))
        h1 = np.vstack((h, 1))
        z = sigmoid(np.dot(W, h1))
        y = y_array[int(y_train[i])]
        total_loss += loss(z, y)
    return float(total_loss / TRAINING_SIZE)

def loss(z, y):
    l = 0
    for i in range(26):
        zi, yi = z[i], y[i]
        l -= yi*np.log(zi) + (1-yi)*np.log(1-zi)
    return l

def plot_loss(loss_array):
    iterations = [i + 1 for i in range(len(loss_array))]
    plt.figure()
    plt.plot(iterations, loss_array)
    plt.title("Average Training Loss vs. Iterations")
    plt.ylabel("Loss")
    plt.xlabel("Number of Iterations")
    plt.show()

def save_classified_images():
    right = 0
    wrong = 0
    index = 0
    while right < 5 or wrong < 5:
        sample = X_validation[index]
        guess = classify(sample)
        sample = sample[:784].reshape((28, 28))
        label = y_validation[index]
        if guess == label:
            if right < 5:
                s = 'right' + str(right) + '.png'
                scipy.misc.imsave(s, sample)
                right += 1
                print(s, guess)
        else:
            if wrong < 5:
                s = 'wrong' + str(wrong) + '.png'
                scipy.misc.imsave(s, sample)
                wrong += 1
                print(s, guess, label)
        index += 1

def classify_test_data():
    TEST_SIZE = testdata.shape[0]
    guesses = []
    for i in range(TEST_SIZE):
        sample = testdata[i]
        guesses.append(classify(sample))
    with open('submission_1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        i = 1
        for g in guesses:
            writer.writerow([i, g])
            i += 1


loss_array = []
for epoch in range(5):
    for i in range(0, TRAINING_SIZE, 40):
        x = np.transpose(X_train[i:i+40])
        h = np.tanh(np.dot(V, x))
        h1 = np.vstack((h, np.ones(40)))
        z = sigmoid(np.dot(W, h1))
        y = []
        for j in range(40):
            y.append(y_array[int(y_train[i+j])].reshape(26))
        y = np.transpose(y)
        grad_w = np.dot(z-y, np.transpose(h1))
        temp1 = np.ones((200, 40)) - h * h
        temp2 = np.dot(np.transpose(W)[:200], z - y)
        grad_v = np.dot(temp1 * temp2, np.transpose(x))
        V = V - grad_v * EPSILON
        W = W - grad_w * EPSILON
    EPSILON *= decay_factor
    print("Epochs:", epoch + 1)
    training_loss = calculate_training_loss()
    print("Training loss:", training_loss)
    loss_array.append(training_loss)
    sys.stdout.flush()

print("Training accuracy:", calculate_training_accuracy())
print("Validation accuracy:", caclulate_validation_accuracy())

plot_loss(loss_array)
classify_test_data()
save_classified_images()