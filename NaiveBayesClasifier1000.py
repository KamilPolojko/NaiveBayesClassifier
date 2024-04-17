import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class NaiveBayesClassifier1000(BaseEstimator, ClassifierMixin):
    def __init__(self, n_bins=10, alpha=1, laplace=False, test = False):
        self.n_bins = n_bins
        self.alpha = alpha
        self.laplace = laplace
        self.classes_ = None
        self.classes_counts_ = None
        self.likelyhood = []
        self.uniqued = []
        self.test = test
        self.dictionary = {}
        self.dictionary_uniqued = {}

    def fit(self, X, y):
        # n_samples=wiersze   n_features=kolumny
        n_samples, n_features = X.shape

        self.classes_, self.classes_counts_ = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)

        self._prior = np.zeros(n_classes)

        for idx in range(len(self.classes_)):
            self._prior[idx] = self.classes_counts_[idx] / self.classes_counts_.sum()

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
                    self.likelyhood[i][j][k] = self.likelyhood[i][j][k]/self.classes_counts_[k]


        for i in range(0, 990):
            key = float(i)
            self.dictionary[key] = {'e': self.likelyhood[i][:][:, 0], 'p': self.likelyhood[i][:][:, 1]}


        for i in range(0, 990):
            key = float(i)
            self.dictionary_uniqued[key] = self.uniqued[i]


        # self.dictionary = {
        #     0.0: {'e': self.likelyhood[0][:][:,0], 'p': self.likelyhood[0][:][:,1]},
        #     1.0: {'e': self.likelyhood[1][:][:,0], 'p': self.likelyhood[1][:][:,1]},
        #     2.0: {'e': self.likelyhood[2][:][:,0], 'p': self.likelyhood[2][:][:,1]},
        #     3.0: {'e': self.likelyhood[3][:][:,0], 'p': self.likelyhood[3][:][:,1]},
        #     4.0: {'e': self.likelyhood[4][:][:,0], 'p': self.likelyhood[4][:][:,1]},
        #     5.0: {'e': self.likelyhood[5][:][:,0], 'p': self.likelyhood[5][:][:,1]},
        #     6.0: {'e': self.likelyhood[6][:][:,0], 'p': self.likelyhood[6][:][:,1]},
        #     7.0: {'e': self.likelyhood[7][:][:,0], 'p': self.likelyhood[7][:][:,1]},
        #     8.0: {'e': self.likelyhood[8][:][:,0], 'p': self.likelyhood[8][:][:,1]},
        #     9.0: {'e': self.likelyhood[9][:][:,0], 'p': self.likelyhood[9][:][:,1]},
        #     10.0: {'e': self.likelyhood[10][:][:,0], 'p': self.likelyhood[10][:][:,1]},
        #     11.0: {'e': self.likelyhood[11][:][:,0], 'p': self.likelyhood[11][:][:,1]},
        #     12.0: {'e': self.likelyhood[12][:][:,0], 'p': self.likelyhood[12][:][:,1]},
        #     13.0: {'e': self.likelyhood[13][:][:,0], 'p': self.likelyhood[13][:][:,1]},
        #     14.0: {'e': self.likelyhood[14][:][:,0], 'p': self.likelyhood[14][:][:,1]},
        #     15.0: {'e': self.likelyhood[15][:][:,0], 'p': self.likelyhood[15][:][:,1]},
        #     16.0: {'e': self.likelyhood[16][:][:,0], 'p': self.likelyhood[16][:][:,1]},
        #     17.0: {'e': self.likelyhood[17][:][:,0], 'p': self.likelyhood[17][:][:,1]},
        #     18.0: {'e': self.likelyhood[18][:][:,0], 'p': self.likelyhood[18][:][:,1]},
        #     19.0: {'e': self.likelyhood[19][:][:,0], 'p': self.likelyhood[19][:][:,1]},
        #     20.0: {'e': self.likelyhood[20][:][:,0], 'p': self.likelyhood[20][:][:,1]},
        #     21.0: {'e': self.likelyhood[21][:][:,0], 'p': self.likelyhood[21][:][:,1]},
        #
        #
        #
        # }


        # self.dictionary_uniqued = {
        #     0.0: self.uniqued[0],
        #     1.0: self.uniqued[1],
        #     2.0: self.uniqued[2],
        #     3.0: self.uniqued[3],
        #     4.0: self.uniqued[4],
        #     5.0: self.uniqued[5],
        #     6.0: self.uniqued[6],
        #     7.0: self.uniqued[7],
        #     8.0: self.uniqued[8],
        #     9.0: self.uniqued[9],
        #     10.0: self.uniqued[10],
        #     11.0: self.uniqued[11],
        #     12.0: self.uniqued[12],
        #     13.0: self.uniqued[13],
        #     14.0: self.uniqued[14],
        #     15.0: self.uniqued[15],
        #     16.0: self.uniqued[16],
        #     17.0: self.uniqued[17],
        #     18.0: self.uniqued[18],
        #     19.0: self.uniqued[19],
        #     20.0: self.uniqued[20],
        #     21.0: self.uniqued[21]
        # }



    def predict(self, X):
        test = self.predict_proba(X)
        list = []

        for i in range(len(test)):
            list.append(np.argmax(test[i]))


        list_final = []
        for i in range(len(test)):
            if(list[i]==0):
                list_final.append('e')
            else:
                list_final.append('p')

        return list_final


    def znajdz_wartosc(self,tablica, szukana_wartosc):
        indeksy = np.where(tablica == szukana_wartosc)
        if len(indeksy[0]) > 0:
            return indeksy[0][0]
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
                        list_1.append(self.dictionary[j][klasa][int(w)])
                        continue
                    else:
                        list_1.append(0)
                        continue
            scores.append(list_1)
            list_1=[]
        return scores

    def obl_praw(self,X,sc,indeks):
        n_samples, n_features = X.shape
        listt = np.ones((n_samples))
        for i in range(n_samples):
            for j in range(n_features):
                if self.test == True:
                    listt[i] += np.log(sc[i][j])
                elif self.test == False:
                    listt[i] *= sc[i][j]
            if self.test == True:
                listt[i] += np.log(self._prior[indeks])
            elif self.test == False:
                listt[i] *= self._prior[indeks]

        return listt

    def predict_proba(self, X):
        n_samples, n_features = X.shape
        lenght = len(self.classes_)

        sc1 = self.get_prawd(X,self.classes_[0])
        sc2 = self.get_prawd(X,self.classes_[1])


        prawd_1 = self.obl_praw(X,sc1,0)
        prawd_2 = self.obl_praw(X,sc2,1)



        list1 =[]
        final = []
        for i in range(n_samples):
            list1.append(prawd_1[i])
            list1.append(prawd_2[i])
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

