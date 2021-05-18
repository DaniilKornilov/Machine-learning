import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# class -1
x1_1 = 17
x2_1 = 7
D_1 = 3
S_1 = np.sqrt(D_1)
N1 = 80

array_11 = np.random.normal(x1_1, S_1, N1).reshape(-1, 1)
array_12 = np.random.normal(x2_1, S_1, N1).reshape(-1, 1)

X1 = np.concatenate((array_11, array_12), axis=1)
Y1 = np.full(len(X1), -1).reshape(-1, 1)

# class 1
x1_2 = 13
x2_2 = 4
D_2 = 3
S_2 = np.sqrt(D_2)
N2 = 20

array_21 = np.random.normal(x1_2, S_2, N2).reshape(-1, 1)
array_22 = np.random.normal(x2_2, S_2, N2).reshape(-1, 1)

X2 = np.concatenate((array_21, array_22), axis=1)
Y2 = np.full(len(X2), 1).reshape(-1, 1)

Y = np.concatenate((Y1, Y2), axis=0)
X = np.concatenate((X1, X2), axis=0)

data = np.concatenate((X, Y), axis=1)
data_frame = pd.DataFrame(data, columns=['X1', 'X2', 'Class'])

# learning
x_train, x_test, y_train, y_test = train_test_split(data_frame.iloc[:, :-1], data_frame['Class'], train_size=0.6)
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_predicted = gnb.predict(x_test)
accuracy = accuracy_score(y_test, y_predicted)
print(accuracy)

# confusion matrix
confusion_m = confusion_matrix(y_test, y_predicted)
disp = plot_confusion_matrix(gnb, x_test, y_test, cmap=plt.cm.Blues)
print(disp.confusion_matrix)
print(confusion_m)

# Raw data chart
plt.figure(figsize=(5, 5))
plt.scatter(X1[:, 0], X1[:, 1])
plt.scatter(X2[:, 0], X2[:, 1])
plt.xlabel('X1')
plt.ylabel('X2')
legend = ('Class -1', 'Class 1')
plt.legend(legend)
plt.grid(True)
plt.show()

# roc curve
pred_prob = gnb.predict_proba(x_test)
fpr, tpr, _ = roc_curve(y_test, pred_prob[:, 1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
legend = ('ROC', 'Random guess')
plt.legend(legend)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# pr curve
precision, recall, thresholds = precision_recall_curve(y_test, pred_prob[:, 1])
plt.plot(recall, precision)
legend = ('PR-Curve', '')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.legend(legend)
plt.show()
