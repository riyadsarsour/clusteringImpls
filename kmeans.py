from math import sqrt, inf
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# contains  longitude of the place is -112.1, and the latitude is 33.5.
PLACES = 'places.txt'
K = 3


def read_data():
    """RETURN LIST OF DATA POINTS, EACH ITEM CONTAINS [LONG,LAT]"""
    data = []
    with open(PLACES, 'r') as f:
        for line in f:
            data.append([float(i) for i in line.split(",")])

    return data


# SHOULD LOOK INTO K-MEANS++
def init_centeroids(data):
    """TAKES RANDOM SAMPLE TO INITIALIZE CLUSTER"""
    return random.sample(data, K)


def euclidean_distance(centeroid, point):
    """CALCULATE EUCLIDEAN DIST FOR DATA POINT TO CENTEROID"""
    distance = 0.0
    # go through each x for centroi d and point then repeat for y
    for i in range(len(centeroid)):
        distance += (centeroid[i]-point[i])**2
    return sqrt(distance)


def gen_clusters(centeroids, data_points):
    """ASSIGN EACH DATAPOINT TO NEAREST CLUSTER"""

    clusters = {}
    for c in centeroids:
        key = tuple(c)
        clusters[key] = []
    # EVAL EACH POINT AGAINST EACH CENTROID
    for point in data_points:
        dist = inf
        cluster = tuple()
        for centeroid in clusters:
            d = euclidean_distance(centeroid, point)
            if d < dist:
                dist = d
                cluster = centeroid

        clusters[cluster].append(point)

    return clusters


def calc_centeroids(clusters):
    """RECALCULATE CENTEROIDS"""
    centeroids = []
    print(clusters.keys())
    for cluster in clusters:
        points = clusters[cluster]

        center = [sum(val[i] for val in points) for i in range(len(points[0]))]
        center = [x/len(points) for x in center]
        centeroids.append(center)

    print(centeroids)
    return centeroids


def main():
    # parse input from location
    data_points = read_data()
    # N data points
    N = len(data_points)
    # init centroids
    centeroids = init_centeroids(data_points)

    # BEGIN CLUSTERING UNTIL CENTEROIDS DO NOT CHANGE

    runs = 1
    while True:
        clusters = gen_clusters(centeroids, data_points)

        new_centeroids = calc_centeroids(clusters)

        if new_centeroids == centeroids:
            break
        # ELSE REOGRANIZE
        centeroids = new_centeroids
        re_calculated_clusters = {}
        for center in centeroids:
            re_calculated_clusters[tuple(center)] = []

        for cluster in clusters:
            for c in re_calculated_clusters:
                re_calculated_clusters[c] = clusters[cluster]
        # Clusters will now have new keys for recalculated centroids
        clusters = re_calculated_clusters
        runs += 1

    # format of OUT file is location_id cluster_label.
    with open('./clusters.txt', 'w') as f:
        for i in range(N):
            index = 0
            for key in clusters:
                if data_points[i] in clusters[key]:
                    f.write(f"{i} {index}\n")
                else:
                    index += 1


if __name__ == "__main__":
    main()
