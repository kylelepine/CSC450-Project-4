import pandas as pd
import numpy as np
from timeit import default_timer as timer
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
            self.k = int((len(self.training_dataset.index) ** (1/2)) /3)

    def set_training_dataset(self, training_data):
        self.training_dataset = training_data

    # allows to reset k 
    def set_k(self, val):
        self.k = val

    # K Nearest Neighbors algorithm
    def classify(self, sample_image):
        classification = None
        neighbors = []
        heapify(neighbors)

        for row_index in range(1, len(self.training_dataset)):
            classification = self.training_dataset.iloc[row_index][0]
            image = self.training_dataset.iloc[row_index][1:].to_numpy()
            sum_euclidean = self.euclidean_distance(image, sample_image)
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

    def euclidean_distance(self, first_coordinates, second_coordinates):
        distance = np.linalg.norm(first_coordinates - second_coordinates)
        return distance

def classify_knn(testing_dataset, training_dataset):
    knn = KNeighborsClassifier(training_dataset)

    total = 0.0
    correct = 0.0
    print(f"k = {knn.k}")
    start =  timer()
    for test_index in range(0, len(testing_dataset.index)):
        real_classification = testing_dataset.iloc[test_index][0]
        data = testing_dataset.iloc[test_index][1:].to_numpy()
        classification = knn.classify((data))
        print(f"Desired class: {real_classification} computed class: {classification}")
        if real_classification == classification: 
            correct += 1.0
            total += 1.0
        else:
            total += 1.0
    end = timer()
    print(f"Accuracy rate: {correct/total * 100}%")
    print(f"Number of misclassified test samples: {total-correct}")
    print(f"Total number of test samples: {total}")
    print(f"Total time elapsed: {end-start}")
    
def main():
    # testing_dataset = 
    # training_dataset = 
    classify_knn(testing_dataset, training_dataset)

if __name__ == '__main__':
    main()