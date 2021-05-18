import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = read_csv('longley.csv', delimiter=',')
Y = data.iloc[:, -1]
np_data = data.to_numpy()
X = np.delete(np_data, [len(np_data[0]) - 1, 4], axis=1)
clf = LinearRegression()
tr_accs = []
t_accs = []
for i in range(1):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
    clf.fit(x_train, y_train)
    print(clf.predict(x_test))
    # print(clf.score(x_train, y_train))
    t_accs.append(clf.score(x_test, y_test))
    tr_accs.append(clf.score(x_train, y_train))
print('Train accuracy:', np.mean(tr_accs))
print('Test accuracy:', np.mean(t_accs))


def lam(i):
    return 10 ** (-3 + 0.2 * i)


ridge = Ridge()

lams = [lam(i) for i in range(26)]

res_test = []
res_train = []
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
for l in lams:
    ridge.alpha = l
    ridge.fit(x_train, y_train)
    res_test.append(mean_squared_error(y_test, ridge.predict(x_test)))
    res_train.append(mean_squared_error(y_train, ridge.predict(x_train)))

plt.plot(lams, res_test)
plt.plot(lams, res_train)
plt.legend(['test', 'train'], loc='best')
plt.xlabel('i')
plt.ylabel('MSE')

plt.show()
