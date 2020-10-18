import pandas as pd
import numpy as np
from cv2 import cv2
# from timeit import default_timer as timer

# Allows the use of a max heap in python.
# This is the predicted optimal datastrucutre to contain the neighbors calculated Euclidean distance measure as
# we would only extract the highest value O(1), insert a new value (log(n)) or extract the max value (log(n)).
# NOTE: Heapq is noramlly implements a heap as a min heap, to use it as a max heap, 
# all values will be multiplied by -1
from heapq import heappop, heappush, heapify

class distance_and_class:
    def __init__(self, distance, classification):
        self.distance = distance
        self.classification = classification
    
    # allows heapq to compare the distance_and_class object to another
    def __lt__(self, other):
        return self.distance < other.distance

# height = 28, width = 28,
class KNeighborsClassifier:
    def __init__(self, training_dataset, k = None):
        self.training_dataset = training_dataset

        # value for k will be set to the square root of the size of the training dataset unless specified
        if k is not None:
            self.k = k
        else:
            self.k = 5

    def set_training_dataset(self, training_data):
        self.training_dataset = training_data

    # allows to reset k 
    def set_k(self, val):
        self.k = val

    # K Nearest Neighbors algorithm
    def classify(self, image):
        classification = None
        neighbors = []
        heapify(neighbors)

        for row_index in range(0, len(self.training_dataset)):
            classification = self.training_dataset.iloc[row_index][0]
            training_image = self.training_dataset.iloc[row_index][1]
            sum_euclidean = self.euclidean_distance(image, training_image)
            DistanceAndClass = distance_and_class(-1 * sum_euclidean, classification)
            if len(neighbors) < self.k:
                heappush(neighbors, DistanceAndClass)
            if neighbors:
                furthest_neighbor = neighbors[0].distance * -1
                if (-1 * DistanceAndClass.distance < furthest_neighbor):
                    heappop(neighbors)
                    heappush(neighbors, DistanceAndClass)
        # TODO: Implement weighted voting
        classifiers = {}
        for neighbor in neighbors:
            neighbor_class = neighbor.classification
            weighted_vote = 1.0 / neighbor.distance * -1
            if neighbor_class in classifiers:
                classifiers[neighbor_class] += weighted_vote
            else:
                classifiers[neighbor_class] = weighted_vote
        classification = max(classifiers, key = classifiers.get)

        return classification

    def euclidean_distance(self, source, target):
        
        if target.shape[0] < source.shape[0]:
            target = np.vstack((target, np.zeros([source.shape[0] - target.shape[0], target.shape[1]])))
        elif source.shape[0] < target.shape[0]:
            source = np.vstack((source, np.zeros([target.shape[0] - source.shape[0], source.shape[1]])))

        if target.shape[1] < source.shape[1]:
            target = np.hstack((target, np.zeros([target.shape[0], source.shape[1] - target.shape[1]])))
        elif source.shape[1] < target.shape[1]:
            source = np.hstack((source, np.zeros([source.shape[0], target.shape[1] - source.shape[1]])))
        
        distance = np.linalg.norm(source - target)
        return distance

def classify_knn(frame, training_dataset):
    knn = KNeighborsClassifier(training_dataset)
    # print(f"k = {knn.k}")
    # start =  timer()
    classification = knn.classify(frame)
    # end = timer()
    # print(f"Total time elapsed: {end-start}")
    return classification
    
def main():
    pass

if __name__ == '__main__':
    main()