from itertools import permutations
import csv
import numpy as np

def Gencomb(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        csv_x = [[int(col) for col in row] for row in reader]
        x_new = []
        for r in csv_x:
            y = r[10]
            x= []
            for i in range(0, 5):
                x.append((r[2*i], r[2*i+1]))
            x_permutations = list(permutations(x))
            for permutation in x_permutations:
                row_new = []
                for card in permutation:
                    row_new.append(card[0])
                    row_new.append(card[1])
                row_new.append(y)
                x_new.append(row_new)
        np.savetxt("/home/arpit/Desktop/temp.csv",x_new, delimiter=",")
        # print list(permutations(x))

Gencomb("../Dataset/poker-hand-training-true copy.csv")