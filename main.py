
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

from NaiveBayesClasifier1000 import NaiveBayesClassifier1000
from NaiveBayesClassifier import NaiveBayesClassifier
from NaiveBayesClassifierContinous import NaiveBayesClassifierContinous


# data = np.genfromtxt('wine.data', delimiter=',')
# X = data[:, 1:]
# y = data[:, 0]
# # from sklearn.datasets import load_wine
# # X, y = load_wine(return_X_y=True)
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=62)
#
#
# nb_classifier = NaiveBayesClassifierContinous()
# nb_classifier.fit(X, y)
#
# y_pred = nb_classifier.predict(X)
#
#
# gaussian_nb = GaussianNB()
# gaussian_nb.fit(X, y)
# gaussian_nb_pred = gaussian_nb.predict(X)
#
# accuracyy = np.mean(y_pred == gaussian_nb_pred)
#
#
# print(f"Porownanie z GaussianNB: {accuracyy:.2%}")
#
# # accuracy = np.mean(y_pred == y_test)
# # print(f"Dokładność klasyfikatora: {accuracy:.2%}")
#




# DLA TYSIACA

data2 = np.genfromtxt('agaricus-lepiota.data', delimiter=",", dtype=str)


X = data2[:, 1:]
y = data2[:, 0]

z = np.hstack((X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X))


print(len(z[0]))

X_train, X_test, y_train, y_test = train_test_split(z, y, test_size=0.4, random_state=42)


nb_classifier = NaiveBayesClassifier1000(laplace=True)
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"Dokładność klasyfikatora: {accuracy:.2%}")

# nb_classifierr = NaiveBayesClassifier1000()
# nb_classifierr.fit(X_train, y_train)
#
# y_predd = nb_classifierr.predict(X_test)
#
# accuracyy = np.mean(y_predd == y_test)
# print(f"Dokładność klasyfikatora: {accuracyy:.2%}")
#
# print(accuracy-accuracyy)
# Z LOGARYTMEM

nb_classifier = NaiveBayesClassifier1000(laplace=True, test=True)
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"Dokładność klasyfikatora: {accuracy:.2%}")

# nb_classifierr = NaiveBayesClassifier1000(test=True)
# nb_classifierr.fit(X_train, y_train)
#
# y_predd = nb_classifierr.predict(X_test)
#
# accuracyy = np.mean(y_predd == y_test)
# print(f"Dokładność klasyfikatora: {accuracyy:.2%}")
#
# print(accuracy-accuracyy)

#PIERWOTNY DLA WINA

# data = np.genfromtxt('wine.data', delimiter=',')
# X = data[:, 1:]
# y = data[:, 0]
#
#
# discretizer = KBinsDiscretizer(10, encode='ordinal', strategy='uniform',subsample = None)
# X_discretizer = discretizer.fit_transform(X)
#
#
# X_train, X_test, y_train, y_test = train_test_split(X_discretizer, y, test_size=0.1, random_state=42)
#
#
#
# nb_classifier = NaiveBayesClassifier(laplace=True)
# nb_classifier.fit(X_train, y_train)
#
# y_pred = nb_classifier.predict(X_test)
#
# accuracy = np.mean(y_pred == y_test)
# print(f"Dokładność klasyfikatora: {accuracy:.2%}")
#
# nb_classifierr = NaiveBayesClassifier()
# nb_classifierr.fit(X_train, y_train)
#
# y_predd = nb_classifierr.predict(X_test)
#
# accuracyy = np.mean(y_predd == y_test)
# print(f"Dokładność klasyfikatora: {accuracyy:.2%}")
#
# print(accuracy-accuracyy)