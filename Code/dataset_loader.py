import numpy as np
import csv

def loadData():
    with open('../Dataset/poker-hand-training-true.csv', 'r') as f:
        reader = csv.reader(f)
        csv_x = [[int(col) for col in row] for row in reader]
        return csv_x

def vectorized_result(j, size):
    e = np.zeros(size)
    e[j - 1] = 1
    return e

def encodeBin(arr):
    res = []
    for row in arr:
        i = 0
        temp = []
        temp2 = []
        temp3 = []
        while i < 10:
            temp2.extend(vectorized_result(row[i], 4))
            temp2.extend(vectorized_result(row[i + 1], 13))
            i += 2
        temp = vectorized_result(row[10], 10)
        temp3.append(temp2)
        temp3.append(temp)
        res.append(temp3)
    return np.array(res)

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

