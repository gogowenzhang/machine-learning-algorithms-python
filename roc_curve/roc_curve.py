import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import pandas as pd

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=2, n_samples=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]


def my_roc_curve(probabilities, labels):
    lst = sorted(zip(probabilities, labels), key=lambda x:x[0], reverse=True)
    labels_sorted = [x[1] for x in lst]
    n = len(lst)
    FPR = []
    TPR = []   
    for i in range(n):
        positive_n = sum(labels_sorted[:i])
        negative_n = i - positive_n
        TPR.append(float(positive_n)/sum(labels_sorted))
        FPR.append(float(negative_n)/(n-sum(labels_sorted)))
    return TPR, FPR, range(n)

tpr, fpr, thresholds = my_roc_curve(probabilities, y_test)

# plt.plot(fpr, tpr)
# plt.xlabel("False Positive Rate (1 - Specificity)")
# plt.ylabel("True Positive Rate (Sensitivity, Recall)")
# plt.title("ROC plot of fake data")

# fpr, tpr, thrsholds = roc_curve(y_test, probabilities, pos_label=1)
# plt.plot(fpr, tpr)
# plt.xlabel("False Positive Rate (1 - Specificity)")
# plt.ylabel("True Positive Rate (Sensitivity, Recall)")
# plt.title("ROC plot of fake data")
# plt.show()


df = pd.read_csv('data/loanf.csv')
y = (df['Interest.Rate'] <= 12).values
X = df[['FICO.Score', 'Loan.Length', 'Loan.Amount']].values
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LogisticRegression()
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]
fpr, tpr, thrsholds = roc_curve(y_test, probabilities, pos_label=1)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity, Recall)")
plt.title("ROC plot of fake data")
plt.show()





    

