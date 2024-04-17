import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_bins=10, alpha=1, laplace=False):
        self.n_bins = n_bins
        self.alpha = alpha
        self.laplace = laplace
        self.classes_ = None
        self.classes_counts_ = None
        self.likelyhood = []
        self.uniqued = []
        self.dictionary = {}
        self.dictionary_uniqued = {}

    def fit(self, X, y):
        # n_samples=wiersze   n_features=kolumny
        n_samples, n_features = X.shape

        self.classes_, self.classes_counts_ = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)

        self._prior = np.zeros(n_classes)

        for idx in range(len(self.classes_)):
            self._prior[idx] = np.log(self.classes_counts_[idx] / self.classes_counts_.sum())

        for j in range(n_features):
            v = np.unique(X[:, j])
            self.uniqued.append(v)
            if self.laplace == True:
                testforlikelyhood = np.ones((len(v), len(self.classes_)))
            else:
                testforlikelyhood = np.zeros((len(v), len(self.classes_)))
            for i in range(n_samples):
                testforlikelyhood[v == X[i, j], y[i] == self.classes_] += 1

            self.likelyhood.append(testforlikelyhood)

        for i in range(n_features):
            for j in range(len(self.likelyhood[i])):
                for k in range(len(self.classes_counts_)):
                    if self.likelyhood[i][j][k]/self.classes_counts_[k] ==0:
                        self.likelyhood[i][j][k] = float('-inf')
                    else:
                        self.likelyhood[i][j][k] = np.log(self.likelyhood[i][j][k]/self.classes_counts_[k])





        self.dictionary = {
            0.0: {1.0: self.likelyhood[0][:][:,0], 2.0: self.likelyhood[0][:][:,1], 3.0: self.likelyhood[0][:][:,2]},
            1.0: {1.0: self.likelyhood[1][:][:,0], 2.0: self.likelyhood[1][:][:,1], 3.0: self.likelyhood[1][:][:,2]},
            2.0: {1.0: self.likelyhood[2][:][:,0], 2.0: self.likelyhood[2][:][:,1], 3.0: self.likelyhood[2][:][:,2]},
            3.0: {1.0: self.likelyhood[3][:][:,0], 2.0: self.likelyhood[3][:][:,1], 3.0: self.likelyhood[3][:][:,2]},
            4.0: {1.0: self.likelyhood[4][:][:,0], 2.0: self.likelyhood[4][:][:,1], 3.0: self.likelyhood[4][:][:,2]},
            5.0: {1.0: self.likelyhood[5][:][:,0], 2.0: self.likelyhood[5][:][:,1], 3.0: self.likelyhood[5][:][:,2]},
            6.0: {1.0: self.likelyhood[6][:][:,0], 2.0: self.likelyhood[6][:][:,1], 3.0: self.likelyhood[6][:][:,2]},
            7.0: {1.0: self.likelyhood[7][:][:,0], 2.0: self.likelyhood[7][:][:,1], 3.0: self.likelyhood[7][:][:,2]},
            8.0: {1.0: self.likelyhood[8][:][:,0], 2.0: self.likelyhood[8][:][:,1], 3.0: self.likelyhood[8][:][:,2]},
            9.0: {1.0: self.likelyhood[9][:][:,0], 2.0: self.likelyhood[9][:][:,1], 3.0: self.likelyhood[9][:][:,2]},
            10.0: {1.0: self.likelyhood[10][:][:,0], 2.0: self.likelyhood[10][:][:,1], 3.0: self.likelyhood[10][:][:,2]},
            11.0: {1.0: self.likelyhood[11][:][:,0], 2.0: self.likelyhood[11][:][:,1], 3.0: self.likelyhood[11][:][:,2]},
            12.0: {1.0: self.likelyhood[12][:][:,0], 2.0: self.likelyhood[12][:][:,1], 3.0: self.likelyhood[12][:][:,2]},

        }

        self.dictionary_uniqued = {
            0.0: self.uniqued[0],
            1.0: self.uniqued[1],
            2.0: self.uniqued[2],
            3.0: self.uniqued[3],
            4.0: self.uniqued[4],
            5.0: self.uniqued[5],
            6.0: self.uniqued[6],
            7.0: self.uniqued[7],
            8.0: self.uniqued[8],
            9.0: self.uniqued[9],
            10.0: self.uniqued[10],
            11.0: self.uniqued[11],
            12.0: self.uniqued[12]
        }


    def predict(self, X):
        test = self.predict_proba(X)
        list = []

        for i in range(len(test)):
            list.append(np.argmax(test[i]))

        for i in range(len(test)):
            list[i] += 1

        return list

    def znajdz_wartosc(self,tablica, szukana_wartosc):
        indeksy = np.where(tablica == szukana_wartosc)

        if len(indeksy[0]) > 0:
            return tablica[indeksy[0][0]]
        else:
            return None

    def get_prawd(self,X, klasa):
        n_samples, n_features = X.shape
        scores = []
        list_1 = []
        for i in range(n_samples):
            for j in range(n_features):
                    if np.isin(X[i][j],self.dictionary_uniqued[j]):
                        w =self.znajdz_wartosc(self.dictionary_uniqued[j],X[i][j])
                        list_1.append(self.dictionary[j][int(klasa)][int(w)])
                        continue
                    else:
                        list_1.append(0)
                        continue

            scores.append(list_1)
            list_1=[]
        return scores


    # def get_prawd(self,X, klasa):
    #     n_samples, n_features = X.shape
    #     scores = []
    #     list_1 = []
    #     for i in range(n_samples):
    #         for j in range(n_features):
    #                 if np.isin(X[i][j],self.dictionary_uniqued[X[i][j]]):
    #                     w =self.znajdz_wartosc(self.dictionary_uniqued[X[i][j]],X[i][j])
    #                     list_1.append(self.dictionary[j][int(klasa)][int(w)])
    #                     continue
    #                 else:
    #                     list_1.append(0)
    #                     continue
    #
    #         scores.append(list_1)
    #         list_1=[]
    #     return scores

    def obl_praw(self,X,sc,indeks):
        n_samples, n_features = X.shape
        listt = np.ones((n_samples))
        for i in range(n_samples):
            for j in range(n_features):
                listt[i] += sc[i][j]
            listt[i] += self._prior[indeks]

        return listt

    def predict_proba(self, X):
        n_samples, n_features = X.shape
        lenght = len(self.classes_)

        sc1 = self.get_prawd(X,self.classes_[0])
        sc2 = self.get_prawd(X,self.classes_[1])
        sc3 = self.get_prawd(X,self.classes_[2])

        prawd_1 = self.obl_praw(X,sc1,0)
        prawd_2 = self.obl_praw(X,sc2,1)
        prawd_3 = self.obl_praw(X,sc3,2)

        list1 =[]
        final = []
        for i in range(n_samples):
            list1.append(prawd_1[i])
            list1.append(prawd_2[i])
            list1.append(prawd_3[i])
            final.append(list1)
            list1=[]

        # sums =np.zeros((n_samples))
        # for i in range(n_samples):
        #     for j in range(lenght):
        #         sums[i] += final[i][j]
        #
        # for i in range(n_samples):
        #     for j in range(lenght):
        #         if np.float64(sums[i])==0:
        #             final[i][j] = 0.0
        #         else:
        #          final[i][j] /= np.float64(sums[i])


        return final

