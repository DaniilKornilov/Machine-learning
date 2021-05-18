import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# A)
def find_accuracies_depth(criterion):
    accuracy_depth = []
    for depth in depths:
        dtc = DecisionTreeClassifier(criterion=criterion, max_depth=depth)
        dtc.fit(x_train, y_train)
        accuracy_depth.append(accuracy_score(y_test, dtc.predict(x_test)))
    return accuracy_depth


def find_accuracies_leaf_nodes(criterion):
    accuracy_ln = []
    for ln in leaf_nodes:
        dtc = DecisionTreeClassifier(criterion=criterion, max_leaf_nodes=ln)
        dtc.fit(x_train, y_train)
        accuracy_ln.append(accuracy_score(y_test, dtc.predict(x_test)))
    return accuracy_ln


np_arr = pd.read_csv('glass.csv', delimiter=',').to_numpy()
Y = np_arr[:, -1]
Y = Y.astype('int')

X = np_arr[:, 1:-1]
X = X.astype('float')

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

depths = [_ for _ in np.arange(1, 100)]
leaf_nodes = [_ for _ in np.arange(3, 100)]

legend = ('Entropy', 'Gini')

accuracies_depth = [find_accuracies_depth('entropy'),
                    find_accuracies_depth('gini')]

accuracies_ln = [find_accuracies_leaf_nodes('entropy'),
                 find_accuracies_leaf_nodes('gini')]

plt.figure(figsize=(10, 10))
plt.grid(True)

for accuracy in accuracies_depth:
    plt.plot(depths, accuracy)
    plt.xlabel('Max tree depth')
    plt.ylabel('Accuracy')
plt.legend(legend)
plt.show()

for accuracy in accuracies_ln:
    plt.plot(leaf_nodes, accuracy)
    plt.xlabel('Max leaf nodes')
    plt.ylabel('Accuracy')
plt.legend(legend)
plt.show()

clf = DecisionTreeClassifier(criterion='gini')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))

plt.figure(figsize=(50, 50))
plot_tree(clf, filled=True)
plt.show()

# B)
np_arr = pd.read_csv('spam7.csv', delimiter=',').to_numpy()
Y = np_arr[:, -1]
Y[Y == 'y'] = 1
Y[Y == 'n'] = 0
Y = Y.astype('int')

X = np_arr[:, :-1]
X = X.astype('float')

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

clf = DecisionTreeClassifier(max_depth=3, criterion='gini')
clf.fit(x_train, y_train)
print(clf.get_depth())
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(50, 50))
plot_tree(clf, filled=True)
plt.show()
