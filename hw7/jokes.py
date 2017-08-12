import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import csv
import scipy.misc
import sys

traindatafilename = "hw7_data/joke_data/joke_train"
data = scipy.io.loadmat(traindatafilename)


print(data)

"""
TRAINING_FRACTION = .8

traindata = data['train_x']
trainlabels = data['train_y']

NUM_FEATURES = traindata.shape[1]
NUM_SAMPLES = traindata.shape[0]

testdata = data['test_x']
"""