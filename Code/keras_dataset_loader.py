import numpy as np
import csv

def loadDataTest(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        csv_x = [[int(col) for col in row] for row in reader]
        csv_x = np.asarray(csv_x)
        csv_y = csv_x[:, -1]  # last column (shares)
        csv_x = np.delete(csv_x, (-1), axis=1)  # delete last column from x
        return encodeBinTest(csv_x, csv_y)


def loadDataTrain(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        csv_x = [[int(col) for col in row] for row in reader]
        csv_x = np.asarray(csv_x)
        csv_y = csv_x[:, -1]  # last column (shares)
        csv_x = np.delete(csv_x, (-1), axis=1)  # delete last column from x
        return encodeBin(csv_x, csv_y)

def vectorized_result_x(suit, rank):
    e = np.zeros(17)
    e[suit - 1] = 1
    e[rank + 3] = 1
    return e

def vectorized_result_y(hand):
    e = np.zeros(10)
    e[hand] = 1
    return e

def findX(x):
    res= []
    for i in range(0, 5):
        res.extend(vectorized_result_x(x[2 * i], x[2 * i + 1]))
    return res

def findY(y):
    res = []
    res.extend(vectorized_result_y(y))
    return res


def encodeBinTest(arrX, arrY):
    test_inputs = [findX(x) for x in arrX]
    test_outputs = [findY(y) for y in arrY]
    # print test_data
    return test_inputs, test_outputs


def encodeBin(arrX, arrY):
    training_inputs = [findX(x) for x in arrX]
    # print training_inputs
    training_outputs = [findY(y) for y in arrY]
    # print training_data
    # training_data = np.array(training_data)
    # print training_data
    return training_inputs, training_outputs

