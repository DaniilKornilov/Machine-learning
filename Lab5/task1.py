import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression

data = read_csv('reglab1.txt', delimiter='\t')

X = data[['x', 'y']].values.reshape(-1, 2)
Y = data['z']

x = X[:, 0]
y = X[:, 1]
z = Y

x_pred = np.linspace(min(x), max(x))
y_pred = np.linspace(min(y), max(y))
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

ols = LinearRegression()
model = ols.fit(X, Y)
predicted = model.predict(model_viz)

r2 = model.score(X, Y)

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0, 0, 0, 0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')

ax1.view_init(elev=28, azim=116)
ax2.view_init(elev=4, azim=130)
ax3.view_init(elev=60, azim=165)

fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)

fig.tight_layout()
plt.show()

print("z - variable, accuracy: ", r2)

X = data[['y', 'z']].values.reshape(-1, 2)
Y = data['x']
model = ols.fit(X, Y)
r2 = model.score(X, Y)

print("x - variable, accuracy: ", r2)

X = data[['x', 'z']].values.reshape(-1, 2)
Y = data['y']
model = ols.fit(X, Y)
r2 = model.score(X, Y)

print("y - variable, accuracy: ", r2)
