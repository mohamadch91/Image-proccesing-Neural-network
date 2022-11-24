import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# loading training set features
start_time = datetime.now()
f = open("Datasets/train_set_features.pkl", "rb")
train_set_features2 = pickle.load(f)
f.close()

# reducing feature vector length
features_STDs = np.std(a=train_set_features2, axis=0)
train_set_features = train_set_features2[:, features_STDs > 52.3]

# changing the range of data between 0 and 1
train_set_features = np.divide(train_set_features, train_set_features.max())

# loading training set labels
f = open("Datasets/train_set_labels.pkl", "rb")
train_set_labels = pickle.load(f)
f.close()

# ------------
# loading test set features
f = open("Datasets/test_set_features.pkl", "rb")
test_set_features2 = pickle.load(f)
f.close()

# reducing feature vector length
features_STDs = np.std(a=test_set_features2, axis=0)
test_set_features = test_set_features2[:, features_STDs > 48]

# changing the range of data between 0 and 1
test_set_features = np.divide(test_set_features, test_set_features.max())

# loading test set labels
f = open("Datasets/test_set_labels.pkl", "rb")
test_set_labels = pickle.load(f)
f.close()

# ------------
# preparing our training and test sets - joining datasets and lables
train_set = []
test_set = []

for i in range(len(train_set_features)):
    label = np.array([0, 0, 0, 0])
    label[int(train_set_labels[i])] = 1
    label = label.reshape(4, 1)
    train_set.append((train_set_features[i].reshape(102, 1), label))

for i in range(len(test_set_features)):
    label = np.array([0, 0, 0, 0])
    label[int(test_set_labels[i])] = 1
    label = label.reshape(4, 1)
    test_set.append((test_set_features[i].reshape(102, 1), label))

# shuffle
random.shuffle(train_set)
random.shuffle(test_set)
# train with  200 data
minimize_train_set = train_set[:200]


# define sigmoid function to calculate sigmoid
def sigmoid(x):
    ans = 1 / (1 + np.exp(-x))
    return ans


# create random seed
np.random.seed(1)
# number of first input layer perceptrons
n_x = 102
# perecptron number of first hidden layer
n_h_1 = 150
##perecptron number of second hidden layer
n_h_2 = 60
# last layer
n_y = 4
# make weights and bias for all layers
W1 = np.random.normal(size=(n_h_1, n_x))
b1 = np.zeros((n_h_1, 1))
W2 = np.random.normal(size=(n_h_2, n_h_1))
b2 = np.zeros((n_h_2, 1))
W3 = np.random.normal(size=(n_y, n_h_2))
b3 = np.zeros((n_y, 1))

counter = 0
# feeddorward for all datas
for i in range(len(minimize_train_set)):
    # label and fetures of one input
    reshape_train = minimize_train_set[i][0]
    reshape_train_label = minimize_train_set[i][1]
    # calculate precptron output with activiation sigmoid function
    S1 = sigmoid(W1 @ reshape_train + b1)
    S2 = sigmoid(W2 @ S1 + b2)
    S3 = sigmoid(W3 @ S2 + b3)
    # find  index of maximum  of output
    index = np.where(S3 == np.amax(S3))
    # find index of maximum of input label
    max_index = np.where(reshape_train_label == np.amax(reshape_train_label))
    if index == max_index:
        # if guess is correct counter++
        counter += 1
# print accuracy and time of execution
print("Accuracy is : " + str(counter / 200))
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
