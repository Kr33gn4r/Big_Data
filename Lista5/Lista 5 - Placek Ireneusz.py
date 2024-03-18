import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pandas as pd

from math import floor
from matplotlib import colormaps
from matplotlib.widgets import RadioButtons
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.datasets import fetch_california_housing, fetch_openml

def sklearn_to_df(skdataset):
    df = pd.DataFrame(skdataset.data, columns=skdataset.feature_names)
    df['target'] = skdataset.target
    return df

def standardize_df(df):
    standardized_df = pd.DataFrame()
    for col in df.columns.values:
        if (df[col].dtype == np.int64 or df[col].dtype == np.float64) and col != "target":
            standardized_df[col] = (df[col] - df[col].mean()) / df[col].std()
        else: standardized_df[col] = df[col]
    return standardized_df

def pca_dataframe(df, n):
    pca_df = pd.DataFrame()
    arr = []
    for i in range(len(df.index)):
        row = []
        for col in df.columns.values:
            if (df[col].dtype == np.int64 or df[col].dtype == np.float64) and col != "target":
                row.append(df[col].iloc[i])
        arr.append(row)
    pca = PCA(n_components=n).fit_transform(arr)
    arr = np.array(pca).T
    for col in range(len(arr)):
        pca_df[f"p{col+1}"] = arr[col]
    pca_df['target'] = df['target']
    return pca_df

def lda_dataframe(df, n):
    lda_df = pd.DataFrame()
    arr = []
    for i in range(len(df.index)):
        row = []
        for col in df.columns.values:
            if (df[col].dtype == np.int64 or df[col].dtype == np.float64) and col != "target":
                row.append(df[col].iloc[i])
        arr.append(row)
    lda = LinearDiscriminantAnalysis(n_components=n, solver="eigen").fit(arr, df['target']).transform(arr)
    arr = np.array(lda).T
    for col in range(len(arr)):
        lda_df[f"p{col + 1}"] = arr[col]
    lda_df['target'] = df['target']
    return lda_df

def svd_dataframe(df, n):
    svd_df = pd.DataFrame()
    arr = []
    for i in range(len(df.index)):
        row = []
        for col in df.columns.values:
            if (df[col].dtype == np.int64 or df[col].dtype == np.float64) and col != "target":
                row.append(df[col].iloc[i])
        arr.append(row)
    svd = TruncatedSVD(n_components=n).fit_transform(arr)
    arr = np.array(svd).T
    for col in range(len(arr)):
        svd_df[f"p{col + 1}"] = arr[col]
    svd_df['target'] = df['target']
    return svd_df

def plot_df(df, ax):
    unique = sorted(df['target'].unique())
    c = np.linspace(0, 1, len(unique))
    cmap = colormaps['plasma']
    color = [cmap(c[i]) for i in range(len(unique))]

    for count, val in enumerate(unique):
        ax.scatter(df['p1'].loc[df['target'] == val], df['p2'].loc[df['target'] == val], color=color[count], label=val)
    ax.legend()

def configure_sets(df, n):
    df_st = standardize_df(df)
    pca_df = pca_dataframe(df, n)
    pca_df_st = pca_dataframe(df_st, n)
    lda_df = lda_dataframe(df, n)
    lda_df_st = lda_dataframe(df_st, n)
    svd_df = svd_dataframe(df, n)
    svd_df_st = svd_dataframe(df_st, n)
    return [[pca_df, lda_df, svd_df], [pca_df_st, lda_df_st, svd_df_st]]

axes = []
def plot_all(df, n, tytul):
    global axes, fig
    fig.suptitle(tytul)
    dfs = configure_sets(df, n)
    subfigs = fig.subfigures(nrows=2, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(["Niestandaryzowane","Standaryzowane"][row])
        axs = subfig.subplots(nrows=1, ncols=3)
        for col, ax in enumerate(axs):
            axes.append(ax)
            plot_df(dfs[row][col], ax)
            ax.set_title(["PCA", "LDA", "SVD"][col])

def radio_change(label):
    global axes, fig
    for ax in axes:
        ax.clear()
    radio_dict = {"Iris": df_iris,
                  "Wine": df_wine,
                  "Diabetes": df_diab,
                  "CH": df_ch,
                  "Open": df_open}
    plot_all(radio_dict[label], 2, label)
    fig.canvas.draw()

df_iris = sklearn_to_df(load_iris())
df_wine = sklearn_to_df(load_wine())
df_diab = sklearn_to_df(load_diabetes())
df_ch = sklearn_to_df(fetch_california_housing())
df_open = sklearn_to_df(fetch_openml(data_id=377))

# zmniejszenie ilości klas z kilku tysięcy
for i in range(len(df_ch.index)):
    val = df_ch['target'].iloc[i]
    df_ch['target'].iloc[i] = f"{(round(val) + floor(val)) / 2}"

for i in range(len(df_diab.index)):
    val = df_diab['target'].iloc[i]
    df_diab['target'].iloc[i] = f"{round(val/20) * 20}".zfill(3)

matplotlib.style.use('ggplot')
fig = plt.figure()
fig.subplots_adjust(left=0.10)

plot_all(df_iris, 2, "Iris")
rax = fig.add_axes([0.01, 0.5, 0.15, 0.15])
radio = RadioButtons(rax, ('Iris', 'Wine', 'Diabetes', 'CH', 'Open'))
radio.on_clicked(radio_change)

plt.show()
