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
                    row_new.append(int(card[0]))
                    row_new.append(int(card[1]))
                row_new.append(int(y))
                x_new.append(row_new)
        np.savetxt("../Dataset/poker-hand-training-true permutation.csv",x_new, delimiter=",", fmt='%i')
        # print list(permutations(x))

def RemoveDuplicate(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        csv_x = [[int(col) for col in row] for row in reader]
        new_rows = []
        for row in csv_x:
            if row not in new_rows:
                new_rows.append(row)
            else:
                print row
        np.savetxt("../Dataset/poker-hand-training-true permutation new.csv", new_rows, delimiter=",", fmt='%i')

# Gencomb("../Dataset/poker-hand-training-true copy.csv")
RemoveDuplicate("../Dataset/poker-hand-training-true permutation.csv")