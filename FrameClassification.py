import numpy as np
from timeit import default_timer as timer

# Allows the use of a max heap in python.
# This is the predicted optimal datastrucutre to contain the neighbors calculated Euclidean distance measure as
# we would only extract the highest value O(1), insert a new value (log(n)) or extract the max value (log(n)).
# NOTE: Heapq is noramlly implements a heap as a min heap, to use it as a max heap, 
# all values will be multiplied by -1
from heapq import heappop, heappush, heapify

class DistanceAndClass:
    def __init__(self, distance, classification):
        self.distance = distance
        self.classification = classification
    
    # allows heapq to compare the DistanceAndClass object to another
    def __lt__(self, other):
        return self.distance < other.distance
    
    def __str__(self):
        return (f"{self.distance},{self.classification}")

# height = 28, width = 28,
class KNeighborsClassifier:
    def __init__(self, trainingDataset, k = None):
        self.training_dataset = trainingDataset

        if k is not None:
            self.k = k
        else:
            self.k = 5

    def set_training_dataset(self, trainingData):
        self.training_dataset = trainingData

    # allows to reset k 
    def set_k(self, val):
        self.k = val

    # K Nearest Neighbors algorithm
    def classify(self, testing_item):
        classification = None
        neighbors = []
        heapify(neighbors)

        for classification_set in self.training_dataset.items():
            classification = classification_set[0]
            for training_item in classification_set[1]:
                sum_euclidean = self.euclidean_distance(testing_item, training_item)
                distance_and_class = DistanceAndClass(-1 * sum_euclidean, classification)
                if len(neighbors) < self.k:
                    heappush(neighbors, distance_and_class)
                elif (-1 * distance_and_class.distance < neighbors[0].distance * -1):
                    heappop(neighbors)
                    heappush(neighbors, distance_and_class)

        classifiers = {}
        for neighbor in neighbors:
            neighbor_class = neighbor.classification
            try:
                weighted_vote = 1.0 / neighbor.distance * -1
            except(ZeroDivisionError) as error:
                print(error)
                weighted_vote = 1.0
            if neighbor_class in classifiers:
                classifiers[neighbor_class] += weighted_vote
            else:
                classifiers[neighbor_class] = weighted_vote
        classification = max(classifiers, key = classifiers.get)

        return classification

    def euclidean_distance(self, source, target):
        
        # if target.shape[0] < source.shape[0]:
        #     target = np.vstack((target, np.zeros([source.shape[0] - target.shape[0], target.shape[1]])))
        # elif source.shape[0] < target.shape[0]:
        #     source = np.vstack((source, np.zeros([target.shape[0] - source.shape[0], source.shape[1]])))

        # if target.shape[1] < source.shape[1]:
        #     target = np.hstack((target, np.zeros([target.shape[0], source.shape[1] - target.shape[1]])))
        # elif source.shape[1] < target.shape[1]:
        #     source = np.hstack((source, np.zeros([source.shape[0], target.shape[1] - source.shape[1]])))
        
        distance = np.linalg.norm(source - target)
        return distance

def classifyKnn(frame, trainingDataset, k = None):
    knn = KNeighborsClassifier(trainingDataset, k)
    # print(f"k = {knn.k}")
    # print(trainingDataset)
    # start =  timer()
    classification = knn.classify(frame)
    # end = timer()
    # print(f"(KNN) classified in: {end-start}")
    return classification