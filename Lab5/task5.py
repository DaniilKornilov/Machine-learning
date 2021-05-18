import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = read_csv('eustock.csv', delimiter=',').to_numpy()
titles = ['DAX', 'SMI', 'CAC', 'FTSE']
columns = []
for i in range(len(data[0])):
    columns.append(data[:, i])
columns = np.array(columns)

x_ticks = [_ for _ in range(1, len(columns[0]) + 1)]

reals = []

plt.figure(figsize=(20, 10))

for column in columns:
    real, = plt.plot(x_ticks, column)

    reals.append(real)

plt.legend(reals, titles)

plt.grid(True)

plt.xticks(())

plt.show()

x_ticks = np.arange(1, len(columns[0]) + 1)

plt.figure(figsize=(20, 10))

for column, title in zip(columns, titles):
    clf = LinearRegression()
    column = column.reshape(-1)
    x_ticks_reshaped = x_ticks.reshape(-1, 1)
    clf.fit(x_ticks_reshaped, column)
    pred = clf.predict(x_ticks_reshaped)
    plt.plot(x_ticks_reshaped, pred)

plt.legend(titles)

plt.grid(True)

plt.xticks(())

plt.show()
