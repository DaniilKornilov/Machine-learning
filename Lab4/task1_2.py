import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

n_list = list(range(1, 101, 10))
n_list = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def get_data(filename):
    np_array = read_csv(filename, delimiter=',').to_numpy()
    x = np_array[:, :-1]
    y = np_array[:, -1]
    return train_test_split(x, y, test_size=0.5)


def get_accuracy(estimator, filename, clf):
    accuracy = {}
    x_train, x_test, y_train, y_test = get_data(filename)
    for n in n_list:
        classifier = clf(base_estimator=estimator, n_estimators=n)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        accuracy[n] = accuracy_score(y_test, y_pred)
    return accuracy


def print_graphics(title, accuracy):
    plt.plot(accuracy.keys(), accuracy.values())
    plt.xlabel('Estimators')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.show()


def view_results_1(filename, classifier):
    print_graphics('Decision tree', get_accuracy(DecisionTreeClassifier(), filename, classifier))
    print_graphics('K neighbours', get_accuracy(KNeighborsClassifier(), filename, classifier))
    print_graphics('Gaussian process', get_accuracy(GaussianProcessClassifier(), filename, classifier))
    print_graphics('SVC', get_accuracy(SVC(), f, c))


def view_results_2(filename, classifier):
    print_graphics('Decision tree', get_accuracy(DecisionTreeClassifier(), filename, classifier))
    print_graphics('Random forest', get_accuracy(RandomForestClassifier(), filename, classifier))
    print_graphics('Gaussian naive', get_accuracy(GaussianNB(), filename, classifier))


f = 'glass.csv'
c = BaggingClassifier
view_results_1(f, c)

# f = 'vehicle.csv'
# c = AdaBoostClassifier
# view_results_2(f, c)
