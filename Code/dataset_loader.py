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
    test_inputs = [np.reshape(findX(x), (85, 1)) for x in arrX]
    test_outputs = [int(y) for y in arrY]
    test_data = np.array([])
    test_data = zip(test_inputs, test_outputs)
    # print test_data
    return test_data


def encodeBin(arrX, arrY):
    training_inputs = [np.reshape(findX(x), (85, 1)) for x in arrX]
    # print training_inputs
    training_outputs = [np.reshape(findY(y), (10, 1)) for y in arrY]
    # print training_outputs
    training_data = np.array([])
    training_data = zip(training_inputs, training_outputs)
    # print training_data
    # training_data = np.array(training_data)
    # print training_data
    return training_data

# loadData()
# def encodeBin(arr):
#
#     main_arr = np.array([])
#     for row in arr:
#         temp = np.zeros(95)
#         counter = 1
#         alternate = 0
#         for num in row:
#             if alternate:
#                 temp[counter+num-2] = 1
#                 alternate = 0
#                 counter+=4
#             else:
#                 temp[counter+num-2] = 1
#                 alternate = 1
#                 counter+=13
#         print temp
#         np.append(main_arr, temp, 0)
#
#     print main_arr

