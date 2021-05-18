from itertools import combinations

import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression


def rss(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred))


data = read_csv('reglab.txt', delimiter='\t')

Y = data['y']
X = data.iloc[:, 1:]

x = [0, 1, 2, 3]
output = sum([list(map(list, combinations(x, i))) for i in range(len(x) + 1)], [])
sec = output[1:]

res = []

labels = np.array(('X1', 'X2', 'X3', 'X4'))

for cols in sec:
    clf = LinearRegression()
    cur_X = X.iloc[:, cols]
    clf.fit(cur_X, Y)
    y_pred = clf.predict(cur_X)
    res.append((cols, rss(Y, y_pred)))

res = sorted(res, key=lambda res: res[1])

for i, r in zip(range(len(res)), res):
    print('{}. Using {} : RSS = {}'.format(i + 1, labels[r[0]], r[1]))
