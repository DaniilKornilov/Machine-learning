import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

data = read_csv('nsw74psid1.csv', delimiter=',').to_numpy()
Y = data[:, -1]
X = np.delete(data, [len(data[0]) - 1], axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

classifiers = [
    DecisionTreeRegressor(),
    LinearRegression(),
    SVR()
]

for clf in classifiers:
    clf.fit(x_train, y_train)
    print('Accuracy of {}: {}%'.format(clf.__class__.__name__, clf.score(x_test, y_test) * 100))
