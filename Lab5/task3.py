import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = read_csv('cygage.txt', delimiter='\t').to_numpy()
Y = data[:, 0]
Y = Y.astype('int')
X = data[:, 1:]

clf = LinearRegression()
accuracies = []
for i in range(50):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    clf.fit(x_train, y_train)
    accuracies.append(clf.score(x_test, y_test))

print('Accuracy:', np.mean(accuracies))

data = read_csv('cygage.txt', delimiter='\t')
Y = data.iloc[:, 0]
X = data.iloc[:, 1:]
plt.scatter(X.iloc[:, 0], Y)
x = X.iloc[:, 0]
y = Y
m, b = np.polyfit(x, y, 1)
plt.plot(x, m * x + b)
plt.show()
