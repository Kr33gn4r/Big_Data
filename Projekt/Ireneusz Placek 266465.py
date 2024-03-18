import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.classification import DecisionTreeClassifier as pysparkDecisionTreeClassifier, \
    RandomForestClassifier as pysparkRandomForestClassifier, \
    MultilayerPerceptronClassifier as pysparkMultilayerPerceptronClassifier
from pyspark.ml.feature import VectorIndexer, VectorAssembler

import tensorflow as tf
import keras.models
from keras import layers

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, \
    f1_score

# import torch nie działa

class NPEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NPEncoder, self).default(obj)

class ScikitLearnClassification():
    def __init__(self, X_train, X_test, Y_train, Y_test, classcol="NSP"):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.classcol = classcol

    def SVM(self):
        clf = SVC()
        clf.fit(self.X_train, self.Y_train)
        Y_classified = clf.predict(self.X_test)
        return self.Y_test.to_numpy(dtype=int), np.array(Y_classified, dtype=int)

    def SGD(self):
        clf = SGDClassifier(loss='log_loss', penalty='elasticnet')
        clf.fit(self.X_train, self.Y_train)
        Y_classified = clf.predict(self.X_test)
        return self.Y_test.to_numpy(dtype=int), np.array(Y_classified, dtype=int)

    def KNeighbors(self):
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(self.X_train, self.Y_train)
        Y_classified = clf.predict(self.X_test)
        return self.Y_test.to_numpy(dtype=int), np.array(Y_classified, dtype=int)

    def DecisionTree(self):
        clf = DecisionTreeClassifier()
        clf.fit(self.X_train, self.Y_train)
        Y_classified = clf.predict(self.X_test)
        return self.Y_test.to_numpy(dtype=int), np.array(Y_classified, dtype=int)

    def RandomForest(self):
        clf = RandomForestClassifier()
        clf.fit(self.X_train, self.Y_train)
        Y_classified = clf.predict(self.X_test)
        return self.Y_test.to_numpy(dtype=int), np.array(Y_classified, dtype=int)

    def MultilayerPerceptron(self):
        clf = MLPClassifier(hidden_layer_sizes=(35, 16, 8, 3), max_iter=1000)
        clf.fit(self.X_train, self.Y_train)
        Y_classified = clf.predict(self.X_test)
        return self.Y_test.to_numpy(dtype=int), np.array(Y_classified, dtype=int)

class SparkClassification():
    def __init__(self, X_train, X_test, Y_train, Y_test, classcol="NSP"):
        self.X_train = X_train.rename(columns={"B":"BE", "E":"EE"})
        self.X_test = X_test.rename(columns={"B":"BE", "E":"EE"})
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.classcol = classcol

        spark = SparkSession.builder.appName("pd to sp").getOrCreate()

        self.data_train = spark.createDataFrame(
            pd.concat([self.X_train, self.Y_train], axis=1))
        self.data_test = spark.createDataFrame(
            pd.concat([self.X_test, self.Y_test], axis=1))
        self.data = spark.createDataFrame(
            pd.concat(
                [pd.concat([self.X_train, self.Y_train], axis=1),
                pd.concat([self.X_test, self.Y_test], axis=1)], axis=0))

        self.vectorAssembler = VectorAssembler(
            inputCols=self.X_train.columns.values.tolist(),
            outputCol="features")
        self.assembled_data = self.vectorAssembler.transform(self.data)
        self.assembled_data_train = self.vectorAssembler.transform(self.data_train)
        self.assembled_data_test = self.vectorAssembler.transform(self.data_test)

        self.featureIndexer = VectorIndexer(
            inputCol="features",
            outputCol="indexedFeatures").fit(self.assembled_data)

    def DecisionTree(self, load=False):
        if load:
            model = PipelineModel.load(
                "/home/rskay/PycharmProjects/PWR/Sem4/Big Data/Projekt/PySpark_DT")
        else:
            dt = pysparkDecisionTreeClassifier(
                labelCol=self.classcol,
                featuresCol="indexedFeatures")

            pipeline = Pipeline(stages=[self.featureIndexer,
                                        dt])
            model = pipeline.fit(self.assembled_data_train)
            model.write().overwrite().save(
                "/home/rskay/PycharmProjects/PWR/Sem4/Big Data/Projekt/PySpark_DT")
        predictions = model.transform(self.assembled_data_test)
        y_true = np.array(predictions.select(self.classcol).collect(), dtype=int).reshape(-1)
        y_pred = np.array(predictions.select("prediction").collect(), dtype=int).reshape(-1)
        return y_true, y_pred

    def RandomForest(self, load=False):
        if load:
            model = PipelineModel.load(
                "/home/rskay/PycharmProjects/PWR/Sem4/Big Data/Projekt/PySpark_RF")
        else:
            rf = pysparkRandomForestClassifier(
                labelCol=self.classcol,
                featuresCol="indexedFeatures")

            pipeline = Pipeline(stages=[self.featureIndexer,
                                        rf])
            model = pipeline.fit(self.assembled_data_train)
            model.write().overwrite().save(
                "/home/rskay/PycharmProjects/PWR/Sem4/Big Data/Projekt/PySpark_RF")
        predictions = model.transform(self.assembled_data_test)
        y_true = np.array(predictions.select(self.classcol).collect(), dtype=int).reshape(-1)
        y_pred = np.array(predictions.select("prediction").collect(), dtype=int).reshape(-1)
        return y_true, y_pred

    def MultilayerPerceptron(self, load=False):
        if load:
            model = PipelineModel.load(
                "/home/rskay/PycharmProjects/PWR/Sem4/Big Data/Projekt/PySpark_MLP")
        else:
            mlp = pysparkMultilayerPerceptronClassifier(
                labelCol=self.classcol,
                featuresCol="indexedFeatures",
                maxIter=1000,
                layers=[35,16,8,4]
            )
            pipeline = Pipeline(stages=[self.featureIndexer,
                                        mlp])
            model = pipeline.fit(self.assembled_data_train)
            model.write().overwrite().save(
                "/home/rskay/PycharmProjects/PWR/Sem4/Big Data/Projekt/PySpark_MLP")
        predictions = model.transform(self.assembled_data_test)
        y_true = np.array(predictions.select(self.classcol).collect(), dtype=int).reshape(-1)
        y_pred = np.array(predictions.select("prediction").collect(), dtype=int).reshape(-1)
        return y_true, y_pred

class TensorFlowClassification():
    def __init__(self, X_train, X_test, Y_train, Y_test, classcol="NSP"):
        self.train = pd.concat([X_train, Y_train], axis=1)
        self.train = self.train.rename(columns={classcol: "target", "B": "BE", "E": "EE"})
        self.train.target = self.train.target.astype(int)
        self.test = pd.concat([X_test, Y_test], axis=1)
        self.test = self.test.rename(columns={classcol: "target", "B": "BE", "E": "EE"})
        self.test.target = self.test.target.astype(int)
        self.train, self.val = train_test_split(self.train, test_size=0.2, random_state=420)

        self.train_ds = self.df_to_dataset(self.train)
        self.val_ds = self.df_to_dataset(self.val, shuffle=False)
        self.test_ds = self.df_to_dataset(self.test, shuffle=False)

    def Classification(self, load=False):
        if load:
            model = keras.models.load_model("//Sem4/Big Data/Projekt/TensorFlow_Model")
        else:
            all_inputs, encoded_features = [], []
            for feature_batch, label_batch in self.train_ds.take(1):
                for header in list(feature_batch.keys()):
                    numeric_col = tf.keras.Input(shape=(1,), name=header)
                    normalization_layer = self.get_normalization_layer(header, self.train_ds)
                    encoded_numeric_col = normalization_layer(numeric_col)
                    all_inputs.append(numeric_col)
                    encoded_features.append(encoded_numeric_col)

            all_features = tf.keras.layers.concatenate(encoded_features)
            x = tf.keras.layers.Dense(35, activation='relu')(all_features)
            x = tf.keras.layers.Dense(16, activation='relu')(x)
            x = tf.keras.layers.Dense(8, activation='relu')(x)
            x = tf.keras.layers.Dropout(.1)(x)
            output = tf.keras.layers.Dense(4, activation='sigmoid')(x)
            model = tf.keras.Model(all_inputs, output)

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.sparse_categorical_crossentropy,
                          metrics=['accuracy'])

            model.fit(self.train_ds,
                      validation_data=self.val_ds,
                      epochs=25)

            model.save("/home/rskay/PycharmProjects/PWR/Sem4/Big Data/Projekt/TensorFlow_Model")

        predict = model.predict(self.test_ds)
        y_pred = np.argmax(predict, 1)
        y_true = Y_test.to_numpy(dtype=int)
        return y_true, y_pred

    def df_to_dataset(self, dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('target')
        df = {key: value[:, tf.newaxis] for key, value in dataframe.items()}
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds

    def get_normalization_layer(self, name, dataset):
        normalizer = layers.Normalization(axis=None)
        feature_ds = dataset.map(lambda x, y: x[name])
        normalizer.adapt(feature_ds)
        return normalizer

def plot(ax, y_true, y_pred, title):
    conf = confusion_matrix(y_true, y_pred, labels=[1,2,3])

    im = ax.imshow(conf, cmap=plt.cm.viridis) # noqa
    ax.set(xticks=[0,1,2],
           yticks=[0,1,2],
           xticklabels=['Normal', 'Suspect', 'Pathologic'],
           yticklabels=['Normal', 'Suspect', 'Pathologic'],
           xlabel='Predykcja',
           ylabel='Wartości prawdziwe',
           title=f'Macierz pomyłek {title}')
    for i in [0,1,2]:
        for j in [0,1,2]:
            text = ax.text(j, i, conf[i, j], ha="center", va="center", color="w")

def res(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {"accuracy": acc,
            "precision": pre,
            "recall": rec,
            "f1": f1}


if __name__ == "__main__":
    # Prepare data
    raw_data = pd.read_excel('/home/rskay/PycharmProjects/PWR/Sem4/Big Data/Projekt/CTG.xls',
                             sheet_name='Raw Data')
    raw_data = raw_data.drop([0, 2127, 2128, 2129])
    raw_data = raw_data.drop(columns=["FileName", "Date", "SegFile"])
    raw_data = raw_data.dropna()
    target = raw_data["NSP"]
    # target = raw_data["CLASS"]
    data = raw_data.drop(columns=["CLASS", "NSP"])

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.25, random_state=2137)

    # Przetworzenie wstepne danych- czasy
    results = {"scikitlearn": {}, "pyspark": {}, "tensorflow": {}}

    t = timer()
    skl = ScikitLearnClassification(X_train, X_test, Y_train, Y_test)
    results["scikitlearn"]["prep_time"] = timer() - t

    t = timer()
    pys = SparkClassification(X_train, X_test, Y_train, Y_test)
    results["pyspark"]["prep_time"] = timer() - t

    t = timer()
    tsf = TensorFlowClassification(X_train, X_test, Y_train, Y_test)
    results["tensorflow"]["prep_time"] = timer() - t

    t = timer()
    y_true, y_pred = skl.SGD()
    et = timer()
    metrics = res(y_true, y_pred)
    results["scikitlearn"]["SGD"] = {"y_true": list(y_true), "y_pred": list(y_pred), "time": et - t, "metrics": metrics}

    t = timer()
    y_true, y_pred = skl.SVM()
    et = timer()
    metrics = res(y_true, y_pred)
    results["scikitlearn"]["SVM"] = {"y_true": list(y_true), "y_pred": list(y_pred), "time": et - t, "metrics": metrics}

    t = timer()
    y_true, y_pred = skl.KNeighbors()
    et = timer()
    metrics = res(y_true, y_pred)
    results["scikitlearn"]["KNeighbors"] = {"y_true": list(y_true), "y_pred": list(y_pred), "time": et - t, "metrics": metrics}

    t = timer()
    y_true, y_pred = skl.DecisionTree()
    et = timer()
    metrics = res(y_true, y_pred)
    results["scikitlearn"]["DecisionTree"] = {"y_true": list(y_true), "y_pred": list(y_pred), "time": et - t, "metrics": metrics}

    t = timer()
    y_true, y_pred = skl.RandomForest()
    et = timer()
    metrics = res(y_true, y_pred)
    results["scikitlearn"]["RandomForest"] = {"y_true": list(y_true), "y_pred": list(y_pred), "time": et - t, "metrics": metrics}

    t = timer()
    y_true, y_pred = skl.MultilayerPerceptron()
    et = timer()
    metrics = res(y_true, y_pred)
    results["scikitlearn"]["MultilayerPerceptron"] = {"y_true": list(y_true), "y_pred": list(y_pred), "time": et - t, "metrics": metrics}

    t = timer()
    y_true, y_pred = pys.DecisionTree()
    et = timer()
    metrics = res(y_true, y_pred)
    results["pyspark"]["DecisionTree"] = {"y_true": list(y_true), "y_pred": list(y_pred), "time": et - t, "metrics": metrics}
    t = timer()
    y_true, y_pred = pys.DecisionTree(load=True)
    et = timer()
    results["pyspark"]["DecisionTree"]["time_load"] = et - t

    t = timer()
    y_true, y_pred = pys.RandomForest() # noqa
    et = timer()
    metrics = res(y_true, y_pred)
    results["pyspark"]["RandomForest"] = {"y_true": list(y_true), "y_pred": list(y_pred), "time": et - t, "metrics": metrics}
    t = timer()
    y_true, y_pred = pys.RandomForest(load=True)
    et = timer()
    results["pyspark"]["RandomForest"]["time_load"] = et - t

    t = timer()
    y_true, y_pred = pys.MultilayerPerceptron() # noqa
    et = timer()
    metrics = res(y_true, y_pred)
    results["pyspark"]["MultilayerPerceptron"] = {"y_true": list(y_true), "y_pred": list(y_pred), "time": et - t, "metrics": metrics}
    t = timer()
    y_true, y_pred = pys.MultilayerPerceptron(load=True)
    et = timer()
    results["pyspark"]["MultilayerPerceptron"]["time_load"] = et - t

    t = timer()
    y_true, y_pred = tsf.Classification() # noqa
    et = timer()
    metrics = res(y_true, y_pred)
    results["tensorflow"]["NeuralNetwork"] = {"y_true": list(y_true), "y_pred": list(y_pred), "time": et - t, "metrics": metrics}
    t = timer()
    y_true, y_pred = tsf.Classification(load=True)
    et = timer()
    results["tensorflow"]["NeuralNetwork"]["time_load"] = et - t

    with open("//Sem4/Big Data/Projekt/3class.json", "w") as f:
        json.dump(results, f, indent=4, cls=NPEncoder)

    fig, ax = plt.subplots(2, 3)
    fig.suptitle("Metody biblioteki ScikitLearn")
    plot(ax[0, 0], results['scikitlearn']['SGD']['y_true'],
         results['scikitlearn']['SGD']['y_pred'], "SGD")
    plot(ax[0, 1], results['scikitlearn']['SVM']['y_true'],
         results['scikitlearn']['SVM']['y_pred'], "SVM")
    plot(ax[0, 2], results['scikitlearn']['KNeighbors']['y_true'],
         results['scikitlearn']['KNeighbors']['y_pred'], "K-Neighbors")
    plot(ax[1, 0], results['scikitlearn']['DecisionTree']['y_true'],
         results['scikitlearn']['DecisionTree']['y_pred'], "Decision Tree")
    plot(ax[1, 1], results['scikitlearn']['RandomForest']['y_true'],
         results['scikitlearn']['RandomForest']['y_pred'], "Random Forest")
    plot(ax[1, 2], results['scikitlearn']['MultilayerPerceptron']['y_true'],
         results['scikitlearn']['MultilayerPerceptron']['y_pred'], "Multilayer Perceptron")
    plt.show()

    fig, ax = plt.subplots(2, 3)
    fig.suptitle("Porównanie metod pomiędzy bibliotekami ScikitLearn i PySpark")
    plot(ax[0, 0], results['scikitlearn']['DecisionTree']['y_true'],
         results['scikitlearn']['DecisionTree']['y_pred'], "ScikitLearn Decision Tree")
    plot(ax[0, 1], results['scikitlearn']['RandomForest']['y_true'],
         results['scikitlearn']['RandomForest']['y_pred'], "ScikitLearn Random Forest")
    plot(ax[0, 2], results['scikitlearn']['MultilayerPerceptron']['y_true'],
         results['scikitlearn']['MultilayerPerceptron']['y_pred'], "ScikitLearn Multilayer Perceptron")
    plot(ax[1, 0], results['pyspark']['DecisionTree']['y_true'],
         results['pyspark']['DecisionTree']['y_pred'], "PySpark Decision Tree")
    plot(ax[1, 1], results['pyspark']['RandomForest']['y_true'],
         results['pyspark']['RandomForest']['y_pred'], "PySpark Random Forest")
    plot(ax[1, 2], results['pyspark']['MultilayerPerceptron']['y_true'],
         results['pyspark']['MultilayerPerceptron']['y_pred'], "PySpark Multilayer Perceptron")
    plt.show()

    fig, ax = plt.subplots(1, 3)
    fig.suptitle("Porównanie metod sieci neuronowych pomiędzy bibliotekami")
    plot(ax[0], results['scikitlearn']['MultilayerPerceptron']['y_true'],
         results['scikitlearn']['MultilayerPerceptron']['y_pred'], "ScikitLearn")
    plot(ax[1], results['pyspark']['MultilayerPerceptron']['y_true'],
         results['pyspark']['MultilayerPerceptron']['y_pred'], "PySpark")
    plot(ax[2], results['tensorflow']['NeuralNetwork']['y_true'],
         results['tensorflow']['NeuralNetwork']['y_pred'], "TensorFlow")
    plt.show()
