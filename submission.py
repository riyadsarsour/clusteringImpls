# Submit this file to Gradescope
from typing import List
from math import sqrt, inf
# you may use other Python standard libraries, but not data
# science libraries, such as numpy, scikit-learn, etc.


class Solution:

    def euclidean_distance(self, point_a, point_b):
        """CALCULATE EUCLIDEAN DIST FOR DATA POINT TO CENTEROID"""
        distance = 0.0
        # go through each x in point_a and point_b then repeat for y
        for i in range(len(point_a)):
            distance += (point_a[i]-point_b[i])**2
        return sqrt(distance)

    def init_clusters(self, X: List[List[float]]):
        """ INITIALIZE ALL POINTS AS CLUSTERS """

        clusters = {}
        for i in range(len(X)):
            clusters[i] = [X[i]]

        return clusters

    def single_link(self, ci, cj):
        """ D(Ci, Cj ) = min{d(vp,vq ) ∣ vp ∈ Ci,vq ∈ Cj } """
        distances = []
        for val_i in ci:
            for val_j in cj:
                distances.append(self.euclidean_distance(val_i, val_j))

        return min(distances)

    def avg_link(self, ci, cj):
        """ D(Ci, Cj ) = mean{d(vp,vq ) ∣ vp ∈ Ci,vq ∈ Cj } """
        distances = []
        for val_i in ci:
            for val_j in cj:
                distances.append(self.euclidean_distance(val_i, val_j))
        return sum(distances) / len(distances)

    def comp_link(self, ci, cj):
        """ D(Ci, Cj ) = max{d(vp,vq ) ∣ vp ∈ Ci,vq ∈ Cj } """

        distances = []
        for val_i in ci:
            for val_j in cj:
                distances.append(self.euclidean_distance(val_i, val_j))
        return max(distances)

    def determine_clusters(self, clusters, M):
        """ Determine the next pair of clusters to merge based on link method """
        min = inf
        closest = ()
        method = None
        if M == 0:
            method = self.single_link
        elif M == 1:
            method = self.avg_link
        elif M == 2:
            method = self.comp_link

        buckets = list(clusters.keys())
        for i in buckets[:-1]:
            for j in buckets[i+1:]:
                distance = method(clusters[i], clusters[j])
                if distance < min:
                    min = distance
                    closest = (i, j)

        return closest

    def merge_clusters(self, clusters, closest):
        """ Combine clusters based on closest clusters on link input """
        ci = closest[0]
        cj = closest[1]
        # merge to cluster
        new_clusters = {0: clusters[ci] + clusters[cj]}

        for cluster in clusters.keys():
            if cluster not in [ci, cj]:
                # place old clusters into new bucket (shrinking in key size so need to reassign)
                new_clusters[len(new_clusters.keys())] = clusters[cluster]
        return new_clusters

    def hclus_single_link(self, X: List[List[float]], K: int) -> List[int]:
        """Single link hierarchical clustering
        Args:
          - X: 2D input data
          - K: the number of output clusters
        Returns:
          A list of integers that represent class labels.
          The number does not matter as long as the clusters are correct.
          For example: [0, 0, 1] is treated the same as [1, 1, 0]"""
        # implement this function
        # init each point as cluster
        clusters = self.init_clusters(X)
        # merge until K clusters left
        buckets = list(clusters.keys())
        while len(clusters.keys()) > K:
            # calc cluster distance
            closest = self.determine_clusters(clusters, 0)
            clusters = self.merge_clusters(clusters, closest)

        labels = []
        for point in X:
            for cluster in clusters:
                if point in clusters[cluster]:
                    labels.append(cluster)
                    break

        return labels

    def hclus_average_link(self, X: List[List[float]], K: int) -> List[int]:
        """Average link hierarchical clustering"""
        # implement this function
        # init each point as cluster
        clusters = self.init_clusters(X)
        while len(clusters.keys()) > K:
            closest = self.determine_clusters(clusters, 1)
            clusters = self.merge_clusters(clusters, closest)

        labels = []
        for point in X:
            for cluster in clusters:
                if point in clusters[cluster]:
                    labels.append(cluster)
                    break

        return labels

    def hclus_complete_link(self, X: List[List[float]], K: int) -> List[int]:
        """Complete link hierarchical clustering"""
        # implement this function
        clusters = self.init_clusters(X)
        while len(clusters.keys()) > K:
            closest = self.determine_clusters(clusters, 2)
            clusters = self.merge_clusters(clusters, closest)

        labels = []
        for point in X:
            for cluster in clusters:
                if point in clusters[cluster]:
                    labels.append(cluster)
                    break

        return labels


# data = []
# # read file
# with open('/Users/riyadsarsour/Downloads/PA-HClus-Handout/sample_test_cases/input02.txt') as f:
#     line_one = f.readline().split(" ")
#     # Number of data points
#     N = int(line_one[0])
#     # K clusters
#     K = int(line_one[1])
#     # Cluster Similarity Measure
#     M = int(line_one[2])

#     for i in range(1, N+1):
#         line = f.readline().strip("\n").split(" ")
#         data.append([float(line[0]), float(line[1])])

#     if M == 0:
#         s = Solution()
#         print(s.hclus_single_link(data, K))
#     if M == 1:
#         s = Solution()
#         print(s.hclus_average_link(data, K))
#     if M == 2:
#         s = Solution()
#         print(s.hclus_average_link(data, K))
