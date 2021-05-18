import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = read_csv('JohnsonJohnson.csv', delimiter=',').to_numpy()

qs = [[] for _ in range(4)]

for i in data:
    qnum = int(i[0][-1])
    qs[qnum - 1].append(i)

for i in range(len(qs)):
    qs[i] = np.array(qs[i])[:, 1].reshape(-1, 1)

qs = np.array(qs)
x_axis = range(len(qs[0]))
years = np.arange(1960, 1981)

all_years = np.sum(np.concatenate(qs, axis=1), axis=1).reshape(-1, 1)

plt.figure(figsize=(20, 10))

for q in qs:
    plt.plot(x_axis, q)

plt.plot(x_axis, all_years)

plt.xticks(x_axis, years)

plt.legend(('Q1', 'Q2', 'Q3', 'Q4', 'Year'))

plt.grid(True)

plt.show()

all_years = np.sum(np.concatenate(qs, axis=1), axis=1).reshape(-1, 1)

plt.figure(figsize=(20, 10))

preds_2016 = []

clf = LinearRegression()
yreshaped = years.reshape(-1, 1)
for q in qs:
    clf.fit(yreshaped, q.reshape(-1))
    pred = clf.predict(yreshaped)
    plt.plot(years, pred)
    preds_2016.append(clf.predict([[2016]])[0])

clf.fit(yreshaped, all_years.reshape(-1))
pred = clf.predict(yreshaped)
plt.plot(years, pred)

plt.xticks(years, [str(i) for i in years])

plt.legend(('Q1', 'Q2', 'Q3', 'Q4', 'Year'))

plt.grid(True)

plt.show()

for p, i in zip(preds_2016, range(1, 5)):
    print('Q' + str(i), p, sep='\t')

clf.fit(yreshaped, all_years)
print('Year', clf.predict([[2016]])[0])

av = np.average(np.concatenate(qs, axis=1), axis=1).reshape(-1, 1)
clf.fit(years.reshape(-1, 1), av)
print('Average ', clf.predict([[2016]])[0])
