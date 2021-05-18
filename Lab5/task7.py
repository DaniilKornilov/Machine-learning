from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.linear_model import LinearRegression

data = read_csv('cars.csv', delimiter=',').to_numpy()

X = data[:, 0]
Y = data[:, 1]
X = X.reshape(-1, 1)

plt.scatter(X, Y)

clf = LinearRegression()
clf.fit(X, Y)
pred = clf.predict(X)

plt.plot(X, pred)
plt.xlabel('Speed')
plt.ylabel('Braking distance')

plt.show()

print('Braking distance:', clf.predict([[40]])[0])
