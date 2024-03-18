import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, load_iris, load_wine, load_digits, \
    make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_curve, auc

from sklearn.cluster import KMeans, MeanShift, OPTICS
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, \
    silhouette_score, calinski_harabasz_score

class Classification:
    def __init__(self, dataset, name:str, labelnames=False, standardize=False, split=0.5):
        try:
            if standardize:
                dataset.data = StandardScaler().fit_transform(dataset.data)
            df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            df['target'] = dataset.target
            if labelnames:
                self.targetnames = dataset.target_names
            else: self.targetnames = sorted(df['target'].unique())
        except Exception:
            df = pd.DataFrame()
            if standardize:
                std = StandardScaler().fit_transform(dataset[0])
                dataset = (std, dataset[1])
            print(dataset[0][0])
            for row in dataset[0]:
                df = pd.concat([df, pd.DataFrame(row.reshape(1, -1))], ignore_index=True)
            df['target'] = dataset[1]
            self.targetnames = sorted(df['target'].unique())
        self.name = name
        self.trainX, self.testX = train_test_split(df, test_size=1-split)

        self.__methods__ = ["kneighbors", "svc", "decisiontree", "randomforest"]

    def __fit_transform__(self, method:str):
        match method.lower():
            case "kneighbors":  clf = KNeighborsClassifier()
            case "svc": clf = SVC(probability=True)
            case "decisiontree": clf = DecisionTreeClassifier()
            case "randomforest": clf = RandomForestClassifier()
            case _: print(f"Error {method}")

        clf.fit(self.trainX.loc[:, self.trainX.columns != 'target'], \
                                   self.trainX['target'])
        predict_labels = clf.predict(self.testX.loc[:, \
                                     self.testX.columns != 'target'])
        predict_probabilities = clf.predict_proba(self.testX.loc[:, \
                                                  self.testX.columns != 'target'])
        return {method.lower(): [predict_labels, predict_probabilities]}

    def fit_all(self):
        self.predict_labels = {method: list(self.__fit_transform__(method).values())[0][0] for method in self.__methods__}
        self.predict_probabilities = {method: list(self.__fit_transform__(method).values())[0][1] for method in self.__methods__}

    def __plot__(self, type: str, ax: plt.Axes, predicted_labels, title, predicted_probabilities=None):
        labels = sorted(self.trainX['target'].unique())
        match type.lower():
            case "confusion":
                confusion_data = confusion_matrix(self.testX['target'], predicted_labels, labels=labels)
                im = ax.imshow(confusion_data, cmap=plt.cm.viridis)

                ax.set(xticks=np.arange(confusion_data.shape[0]),
                       yticks=np.arange(confusion_data.shape[0]),
                       xticklabels=self.targetnames,
                       xlabel='Prediction',
                       ylabel='True values',
                       title=f'{title} Confusion matrix')
                ax.set_yticklabels(self.targetnames, rotation=90, ha="center", rotation_mode="anchor")

                for i in labels:
                    for j in labels:
                        text = ax.text(j, i, confusion_data[i, j], ha="center", va="center", color="w")
            case "roc":
                metrics, fpr, tpr, roc_auc = self.__roc_metrics__(predicted_labels, predicted_probabilities)
                for iter, i in enumerate(labels):
                    ax.plot(fpr[i], tpr[i], label=f'Class \"{self.targetnames[iter]}\" (AUC = {roc_auc[i]:0.4f})')
                ax.set(xlabel='False Positive Rate',
                       ylabel='True Positive Rate',
                       title=f'{title} ROC plot')
                s = [f'{metrics[key]:0.4f} : {key}\n' for key in list(metrics.keys())]
                ax.text(0.8, -0.37, ''.join(s))
                ax.legend()

    def __roc_metrics__(self, predicted_labels, predicted_probabilities):
        metrics = {}
        metrics['Accuracy'] = accuracy_score(self.testX['target'], predicted_labels)
        metrics['Precision'] = precision_score(self.testX['target'], predicted_labels, average='weighted', zero_division=0)
        metrics['Recall'] = recall_score(self.testX['target'], predicted_labels, average='weighted')
        metrics['F1'] = f1_score(self.testX['target'], predicted_labels, average='weighted')

        labels = sorted(self.trainX['target'].unique())
        ytrue = label_binarize(self.testX['target'], classes=labels)
        yscore = predicted_probabilities
        fpr, tpr, roc_auc = {}, {}, {}
        for i, name in enumerate(labels):
            fpr[name], tpr[name], _ = roc_curve(ytrue[:, i], yscore[:, i])
            roc_auc[name] = auc(fpr[i], tpr[i])
        return metrics, fpr, tpr, roc_auc

    def plot_all(self):
        columns = list(self.predict_labels.keys())
        fig, ax = plt.subplots(2, len(columns))
        fig.suptitle(f"{self.name} comparison")

        for iter, col in enumerate(columns):
            self.__plot__('confusion', ax[0, iter], self.predict_labels[col], col)
            self.__plot__('roc', ax[1, iter], self.predict_labels[col], col, self.predict_probabilities[col])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

class Clustering:
    def __init__(self, generated_data, title, n_clusters=2, standardize=False):
        features = np.array(generated_data[0])
        self.labels = np.array(generated_data[1])
        if standardize:
            self.features = (features - features.mean())/(features.std())
        else: self.features = features
        self.title = title

        self.__methods__=['kmeans', 'meanshift', 'optics']
        self.n_clusters = n_clusters

    def __fit__(self, method):
        match method.lower():
            case 'kmeans': clu = KMeans(n_clusters=self.n_clusters)
            case 'meanshift': clu = MeanShift(bandwidth=self.n_clusters)
            case 'optics': clu = OPTICS(min_samples=self.n_clusters)
        predicted_labels = clu.fit_predict(self.features)
        return predicted_labels

    def fit_all(self):
        self.predicted_labels = {method: self.__fit__(method) for method in self.__methods__}

    def __plot__(self, ax:plt.Axes, labels:list, title:str, metricflag=True):
        ax.scatter(self.features[:, 0], self.features[:, 1], c=labels, cmap=plt.cm.viridis, label=list(set(labels)))
        ax.set(xlabel='feature_0',
               ylabel='feature_1',
               title=f'{title} Scatterplot')
        if metricflag:
            metrics = self.__metrics__(labels)
            s = [f'{metrics[key]:0.4f} : {key}\n' for key in list(metrics.keys())]
            ax.text(-1, -1, ''.join(s))

    def plot_all(self):
        columns = list(self.predicted_labels.keys())
        fig, ax = plt.subplots(1, len(columns)+1)
        methods = ['True', *self.__methods__]
        labels = {'True': self.labels, **self.predicted_labels}
        for iter, method in enumerate(methods):
            if method == 'True':
                self.__plot__(ax[iter], labels[method], title=method, metricflag=False)
            else:
                self.__plot__(ax[iter], labels[method], title=method)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

    def __metrics__(self, labels):
        metrics = {}
        try:
            metrics['Silhouette'] = silhouette_score(self.features, labels)
            metrics['CH Index'] = calinski_harabasz_score(self.features, labels)
            metrics['Rand Index'] = adjusted_rand_score(self.labels, labels)
            metrics['Homogeneity'] = homogeneity_score(self.labels, labels)
            metrics['Mutual Inf.'] = adjusted_mutual_info_score(self.labels, labels)
            return metrics
        except Exception: return {"ERROR": 0}


iris = Classification(load_iris(), "Iris dataset", labelnames=True, standardize=True)
iris.fit_all()
iris.plot_all()

wine = Classification(load_wine(), "Wine dataset", labelnames=True)
wine.fit_all()
wine.plot_all()

digits = Classification(load_digits(), "Digits dataset", labelnames=True)
digits.fit_all()
digits.plot_all()

blobs = Clustering(make_blobs(300, 2, centers=5), "Blobs", n_clusters=5)
blobs.fit_all()
blobs.plot_all()

circles = Clustering(make_circles(300, noise=0), "Circles", n_clusters=2)
circles.fit_all()
circles.plot_all()

classification_data = make_classification(n_samples=500, shift=1, scale=2, hypercube=False, n_classes=6, n_informative=4)
classif = Classification(classification_data, "make_classification")
classif_stan = Classification(classification_data, "make_classification Standardized", standardize=True)
classif.fit_all()
classif.plot_all()
classif_stan.fit_all()
classif_stan.plot_all()