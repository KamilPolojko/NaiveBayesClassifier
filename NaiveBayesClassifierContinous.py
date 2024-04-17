import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class NaiveBayesClassifierContinous(BaseEstimator, ClassifierMixin):
    def __init__(self, n_bins=10, alpha=1, test =False):
        self.n_bins = n_bins
        self.alpha = alpha
        self.test = test
        self.classes_ = None
        self.classes_counts_ = None
        self.likelyhood = []
        self.uniqued = []


    def fit(self, X, y):
        # n_samples=wiersze   n_features=kolumny
        n_samples, n_features = X.shape

        self.classes_, self.classes_counts_ = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)

        self._prior = np.zeros(n_classes)

        for idx in range(len(self.classes_)):
            if self.test == True:
                self._prior[idx] = np.log(self.classes_counts_[idx] / self.classes_counts_.sum())

        self._mean = np.zeros((n_features,n_classes))
        self._var = np.zeros((n_features,n_classes))
        self._var_pred = np.zeros((n_features,n_classes))


        for i in range(n_features):
            self._mean[i][0] = np.mean(X[y==self.classes_[0]][:,i])
            self._mean[i][1] = np.mean(X[y==self.classes_[1]][:,i])
            self._mean[i][2] = np.mean(X[y==self.classes_[2]][:,i])



        sum = 0;
        for i in range(n_features):
            for j in range(n_classes):
                X_c =X[y == self.classes_[j]][:, i]
                for z in range(len(X[y==self.classes_[j]][:,i])):
                     sum += np.power(X_c[z] - self._mean[i][j],2)
                self._var_pred[i][j] = sum
                sum=0


        for i in range(n_features):
            for j in range(n_classes):
                self._var[i][j] = np.sqrt((1/(self.classes_counts_[j]-1))*self._var_pred[i][j])





    def predict(self, X):
        test = self.predict_proba(X)
        list = []
        for i in range(len(test)):
            list.append(np.argmax(test[i]))

        for i in range(len(test)):
            list[i] += 1

        return list



    def predict_proba(self, X):
        n_samples, n_features = X.shape
        lenght = len(self.classes_)

        list = []
        list_final = []
        c = 0
        for j in range(n_samples):
            for z in range(lenght):
                for i in range(n_features):
                    # c *= (1/(self._var[i][z]*np.sqrt(2* np.pi))) * np.exp(-1*(np.power(X[j][i] - self._mean[i][z],2)/(2*np.power(self._var[i][z],2))))
                    c += ((-1* np.log(self._var[i][z]))-((np.power(X[j][i]-self._mean[i][z],2)/(2*np.power(self._var[i][z],2)))))
                list.append(c)
                c=0
            list_final.append(list)
            list = []


        for i in range(n_samples):
            for j in range(lenght):
                list_final[i][j] += self._prior[j]

        # sums =np.zeros((n_samples))
        # for i in range(n_samples):
        #     for j in range(lenght):
        #         sums[i] += list_final[i][j]
        #
        # for i in range(n_samples):
        #     for j in range(lenght):
        #         if np.float64(sums[i])==0:
        #             list_final[i][j] = 0.0
        #         else:
        #          list_final[i][j] /= np.float64(sums[i])

        return list_final