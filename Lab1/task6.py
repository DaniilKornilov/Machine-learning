import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

np_arr_train = pd.read_csv('bank_scoring_train.csv', delimiter='\t').to_numpy()
np_arr_test = pd.read_csv('bank_scoring_test.csv', delimiter='\t').to_numpy()


def get_x_y(arr):
    y = arr[:, 0]
    y = y.astype('int')
    x = arr[:, 1:]
    x = x.astype('float')
    return x, y


x_train, y_train = get_x_y(np_arr_train)
x_test, y_test = get_x_y(np_arr_test)

classifiers = (tree.DecisionTreeClassifier(),
               GaussianNB(),
               KNeighborsClassifier(n_neighbors=1),
               svm.SVC(kernel='sigmoid'),
               svm.SVC(kernel='rbf'),
               RandomForestClassifier(),
               AdaBoostClassifier())

titles = ('Decision tree',
          'Naive Gaussian',
          'K neighbours',
          'SVC Sigmoid',
          'SVC RBF',
          'Random forest',
          'Ada boost')

classifiers[0].fit(x_train, y_train)

print('Classifiers: ')
print('-' * 20)
for m, title in zip(classifiers, titles):
    y_pred = m.fit(x_train, y_train).predict(x_test)
    print(title)
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    disp = plot_confusion_matrix(m, x_test, y_test, cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()
    print('Issue a loan:', len(y_test[y_test == 1]), 'Not issue a loan:', len(y_test[y_test == 0]))
    print('-' * 20)
