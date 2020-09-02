#!/usr/bin/env python3

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def read_data(path):
    """
    Args:path to the dataset
    Returns:[ [x_1, ..., x_n], ... ,[x_1, ..., x_n]]
    """
    data_set = []
    with open(path) as data_from_file:
        for element_as_string in data_from_file:
                split_element = element_as_string.split(',')
                feature_vec = [float(i) for i in split_element[:]]
                data_set.append(feature_vec)
    return data_set

def init_centers_random(data_set, k):
    """
    Args:
        data_set: a list of data points, where each data point is a list of features.
        k: the number of mean/clusters.
    Returns:
        centers: a list of k elements: centers initialized using random k data points in your data_set.
                 Each center is a list of numerical values. i.e., 'vals' of a data point.
    """
    random.shuffle(data_set)
    centers = data_set[:k]
    return centers 

def dist(vals, center):
    """
    Args:
        vals: a list of numbers (i.e. 'vals' of a data_point)
        center: a list of numbers, the center of a cluster.
    Returns:
         d: the euclidean distance from a data point to the center of a cluster
    """
    paired_vec = zip(vals, center)
    vec_difference_sq = [(a-b)**2 for a, b in paired_vec]
    magnitude = sum(vec_difference_sq)**0.5
    return magnitude

def get_nearest_center(vals, centers):
    """
    Args:
        vals: a list of numbers (i.e. 'vals' of a data point)
        centers: a list of center points.
    Returns: c_idx: a number, the index of the center of the nearest cluster, to which the given data point is assigned to.
    """
    c_idx = 0 
    smallest_dist = np.inf 
    for index, center in enumerate(centers): 
        curr_dist = dist(vals, center)
        if(curr_dist < smallest_dist):
            smallest_dist = curr_dist
            c_idx = index
    return c_idx

# TODO: compute element-wise addition of two vectors.
def vect_add(x, y):
    """
    Helper function for recalculate_centers: compute the element-wise addition of two lists.
    """
    paired_vec = zip(x, y)
    return [(a+b) for a, b in paired_vec]


# TODO: averaging n vectors.
def vect_avg(s, n):
    """
    Helper function for recalculate_centers: Averaging n lists.
    Args:
        s: a list of numerical values: the element-wise addition over n lists.
        n: a number, number of lists
    Returns:
        s: a list of numerical values: the averaging result of n lists.
    """
    return [element/float(n) for element in s]


def recalculate_centers(clusters, data_set):
    """
    Re-calculate the centers as the mean vector of each cluster.
    Args:
         clusters: a list of clusters. Each cluster is a list of data_points assigned to that cluster.

    Returns:
        centers: a list of new centers as the mean vector of each cluster.
    """
    new_centers = []
    for clust in clusters:
        if(len(clust) == 0):
            random.shuffle(data_set)
            new_centers.append(data_set[0])
        else:
            sum_of_feature_vecs = [0]*len(clust[0])
            for feature_vec in clust:
                sum_of_feature_vecs = vect_add(sum_of_feature_vecs, feature_vec)
            mean = vect_avg(sum_of_feature_vecs, len(clust))
            new_centers.append(mean)
    return new_centers
    

def train_kmean(data_set, centers, iter_limit):
    """
    Args:
        data_set: a list of data points, where each data point is a list of features.
        centers: a list of initial centers.
        iter_limit: a number, iteration limit
    Returns:
        centers: a list of updates centers/mean vectors.
        clusters: a list of clusters. Each cluster is a list of data points.
        num_iterations: a number, num of iteration when converged.
    """
    new_centers = centers
    num_iterations = 0
    clusters = []
    while num_iterations < iter_limit:
        prev_centers = new_centers
        clusters = [[] for i in range(len(centers))]
        num_iterations+=1
        for feature_vec in data_set:
            c_idx = get_nearest_center(feature_vec, new_centers)
            clusters[c_idx].append(feature_vec)
        new_centers = recalculate_centers(clusters, data_set)
        if new_centers == prev_centers:
            break; # no change was made 
    centers = new_centers
    return centers, clusters, num_iterations

# TODO: helper function: compute within group sum of squares
def within_cluster_ss(cluster, center):
    """
    For each cluster, compute the sum of squares of euclidean distance
    from each data point in the cluster to the empirical mean of this cluster.
    Please note that the euclidean distance is squared in this function.

    Args:
        cluster: a list of data points.
        center: the center for the given cluster.

    Returns:
        ss: a number, the within cluster sum of squares.
    """
    ss = 0
    for feature_vec in cluster:
        paired_vec = zip(feature_vec, center)
        euclid_dist_sq = [(a-b)**2 for a, b in paired_vec]
        ss += sum(euclid_dist_sq)
    return ss

def sum_of_within_cluster_ss(clusters, centers):
    """
    For total of k clusters, compute the sum of all k within_group_ss(cluster).
    Args:
        clusters: a list of clusters.
        centers: a list of centers of the given clusters.
    Returns:
        sss: a number, the sum of within cluster sum of squares for all clusters.
    """
    sss = 0
    for i in range(len(clusters)):
        ss = within_cluster_ss(clusters[i], centers[i])
        sss += ss
    return sss

def test():
    ds = read_data('./simple.txt')
    centers = init_centers_random(ds, 3)
    centers, clusters, num_iterations = train_kmean(ds, centers, 20)
    print(new_centers, num_iterations)
    sss = sum_of_within_cluster_ss(clusters, new_centers)
    print(sss)

def winetest():
    ds = read_data('./wine.txt')
    list_of_sss = []
    list_of_ks = []
    for k in range(2,11):
        centers = init_centers_random(ds, k)
        centers, clusters, num_iterations = train_kmean(ds, centers, 50)
        sss = sum_of_within_cluster_ss(clusters, centers)
        list_of_sss.append(sss)
        list_of_ks.append(k)
    plt.xlabel('k values')
    plt.xticks(np.arange(0, 12, step=1))
    plt.ylabel('sum of within_cluster_ss values')
    plt.title('comparing sum of within_cluster_ss as k gets larger')
    plt.scatter(list_of_ks, list_of_sss)
    plt.plot(list_of_ks, list_of_sss, linewidth=0.5)
    plt.show()



