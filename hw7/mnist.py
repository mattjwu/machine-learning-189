import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import csv
import scipy.misc
import sys
import os
import random

traindatafilename = "hw7_data/mnist_data/images"
data = scipy.io.loadmat(traindatafilename)

images = data['images']

samples = np.transpose(images.reshape((784, 60000)))
samples = samples * 1.0

def k_means(num_clusters, max_iterations = 250):
    #starting_indecies = random.sample(range(60000), num_clusters)

    #k_means++ method of picking starting indexes
    first_index = random.randrange(60000) #Pick starting point
    starting_indecies = [first_index]
    current_sample = samples[first_index]

    distances = np.sum((samples - current_sample)**2, axis=1)

    for i in range(num_clusters - 1):
        #Select a random point via kmeans++ method
        total_distance = np.sum(distances)
        target = np.random.random() * total_distance
        val = 0
        j = 0
        while True:
            val += distances[j]
            if val >= target:
                break
            j += 1
        #Add the new point and update point distances
        starting_indecies.append(j)
        current_sample = samples[j]
        dist_to_new_point = np.sum((samples - current_sample)**2, axis=1)
        distances = np.minimum(distances, dist_to_new_point)

    #Initialize the clusters
    clusters = samples[[starting_indecies]]

    iteration = 0

    #samp_norm_squared = np.sum(samples**2, axis = 1).reshape((60000, 1))
    prev_closest_clusters = None

    #Update loop
    while iteration < max_iterations:
        iteration += 1

        #Find points closest to each cluster
        #||u-v||^2 == u*u + v*v - 2(u*v)
        clust_norm_squared = np.sum(clusters ** 2, axis = 1)
        samp_dot_clust = np.inner(samples, clusters)
        #squared_distance_mat = samp_norm_squared + clust_norm_squared - 2 * samp_dot_clust
        squared_distance_mat = clust_norm_squared - 2 * samp_dot_clust

        #squared_distance_mat:
        #(num_samples, num_clusters) matrix
        #squared_distance_mat[i, j] is the distance between point i and cluster j

        #closest_cluster finds the closest cluster index for each sample point
        closest_clusters = np.argmin(squared_distance_mat, axis=1)

        if iteration > 1:
            if (closest_clusters == prev_closest_clusters).all():
                print("\nConverged")
                break

        prev_closest_clusters = closest_clusters

        #cost = 0
        #Calculate the new cluster centers
        for c in range(num_clusters):
            samples_in_cluster = samples[np.where(closest_clusters == c)]
            new_center = np.sum(samples_in_cluster, axis=0) / samples_in_cluster.shape[0]
            #cost += np.sum((samples_in_cluster - new_center)**2)
            clusters[c] = new_center

        print("\nIteration " + str(iteration))
        #print("Cost: " + str(cost))
        sys.stdout.flush()

    if iteration == max_iterations:
        print("\nMaximum iterations reached")

    #Finished finding cluster centers, Save cluster center images
    directory = 'images/mnist-' + str(num_clusters) + '-clusters/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for c in range(num_clusters):
        cluster = clusters[c]
        cluster = cluster.reshape((28, 28))
        s = directory + str(c) + '.png'
        scipy.misc.imsave(s, cluster)

#k_means(5)
#k_means(10)
#k_means(20)