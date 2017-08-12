import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import csv
import scipy.misc
import sys

load_directory = 'hw7_data/low-rank_data/'
save_directory = 'images/low-rank_approx/'

np.set_printoptions(threshold=np.inf)

def make_low_rank_image(filename, rank):
    pic_name = filename[:-4]
    picture = scipy.misc.imread(load_directory + filename)
    u, s, v = np.linalg.svd(picture, False)
    sig = np.zeros((s.shape[0], s.shape[0]))
    for i in range(rank):
        sig[i, i] = s[i]

    new_picture = np.dot(np.dot(u, sig), v)
    new_picture = np.clip(new_picture, 0, 255)
    new_picture = (new_picture + .5).astype('uint8')

    scipy.misc.imsave(save_directory + pic_name + '-' +
        str(rank) + '.png', new_picture)

def plot_MSE(filename):
    picture = scipy.misc.imread(load_directory + filename)
    u, s, v = np.linalg.svd(picture, False)
    sig = np.zeros((s.shape[0], s.shape[0]))
    error_array = []
    for i in range(100):
        sig[i, i] = s[i]
        new_picture = np.dot(np.dot(u, sig), v)
        error = np.sum((picture - new_picture) ** 2)
        error_array.append(error)
    rank_array = [i + 1 for i in range(100)]
    plt.plot(rank_array, error_array)
    plt.title("Mean Squared Error vs. Rank")
    plt.ylabel("Error")
    plt.xlabel("Rank Approximation")
    plt.show()

#make_low_rank_image('face.jpg', 5)
#make_low_rank_image('face.jpg', 20)
#make_low_rank_image('face.jpg', 100)

#make_low_rank_image('sky.jpg', 5)
#make_low_rank_image('sky.jpg', 20)
#make_low_rank_image('sky.jpg', 100)

make_low_rank_image('face.jpg', 80)

#plot_MSE('face.jpg')