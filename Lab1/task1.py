import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def print_graphics(x, y, title):
    accuracy = []
    set_ratio = [_ for _ in np.arange(0.01, 0.99, 0.01)]
    for i in set_ratio:
        clf = GaussianNB()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=i)
        clf.fit(x_train, y_train)
        accuracy.append(accuracy_score(y_test, clf.predict(x_test)))
    plt.plot(set_ratio, accuracy)
    plt.title(title)
    plt.xlabel('Training set / Available set')
    plt.ylabel('Accuracy')
    plt.show()


# tic - tac - toe
np_arr = pd.read_csv('tic_tac_toe.txt', delimiter=',').to_numpy()

Y = np_arr[:, -1]
Y[Y == 'negative'] = 0
Y[Y == 'positive'] = 1
Y = Y.astype('int')

X = np_arr[:, :-1]
X[X == 'o'] = 0
X[X == 'x'] = 1
X[X == 'b'] = 2
X = X.astype('int')

print_graphics(X, Y, 'tic-tac-toe')

# spam
np_arr = pd.read_csv('spam.csv', delimiter=',').to_numpy()

Y = np_arr[:, -1]
Y[Y == 'spam'] = 1
Y[Y == 'nonspam'] = 0
Y = Y.astype('int')

X = np_arr[:, :-1]
X = X.astype('float')

print_graphics(X, Y, 'spam')
