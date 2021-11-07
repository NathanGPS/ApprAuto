import pandas as pd
from sacred import Experiment
from data import get_data, data_ingredient
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from mpl_toolkits import mplot3d
import numpy as np

ex = Experiment("drinkable water", ingredients=[data_ingredient])


@ex.config
def config():
    seed = 2021
    fill_na_method = "mean by class"
    remove_outliers = False
    normalize_features = True
    reduce_dimension = None
    model_used = "Random Forest"


def plot_bar_charts(data, title, ax=None):
    ax = data.plot.bar(ax=ax)
    ax.set_ylim(0, 0.25)

    # set x labels to %
    vals = ax.get_yticks()
    ax.set_yticklabels(['{}%'.format(int(x * 100)) for x in vals])

    ax.set_axisbelow(True)
    ax.grid(color='#cccccc', linestyle='dashed')
    plt.xticks(rotation=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, fontdict={"fontsize": 12}, y=-0.1)

    # plt.show()


def plot_pie_chart(data, column, ax, only_keep_na=False):
    if only_keep_na:
        data = data[data[column].isna()]["Potability"]
    else:
        data = data["Potability"]
    df = pd.Series([data.sum(), len(data) - data.sum()], index=['Potable', 'Not Potable'], name="")
    if only_keep_na:
        plot, = df.plot.pie(subplots=True, legend=False, labels=['', ''], ax=ax)
        plot.set_title("Datapoints where " + column + " is missing", fontdict={"fontsize": 12}, y=-0.1)
    else:
        plot, = df.plot.pie(subplots=True, legend=False, ax=ax, labeldistance=None)
        plot.set_axis_off()
        plot.legend(["Potable", 'Not potable'], loc=(1, 1), fontsize=12)
        plot.set_title("All datapoints", fontdict={"fontsize": 12}, y=-0.1)


@ex.automain
def main(fill_na_method, remove_outliers, normalize_features, reduce_dimension, model_used, seed):
    # --- Setting the seed --- #
    np.random.seed(seed)

    data = get_data()

    # # ignoring dropna for now
    # data.dropna(inplace=True)
    # data.dropna(axis=0, inplace=True)
    if fill_na_method == "mean by class":
        mask = data["Potability"] == 1
        fill_na_potable = data[mask].mean()
        fill_na_non_potable = data[~mask].mean()

        data[mask] = data[mask].fillna(fill_na_potable)
        data[~mask] = data[~mask].fillna(fill_na_non_potable)
    elif fill_na_method == "median by class":
        mask = data["Potability"] == 1
        fill_na_potable = data[mask].median()
        fill_na_non_potable = data[~mask].median()

        data[mask] = data[mask].fillna(fill_na_potable)
        data[~mask] = data[~mask].fillna(fill_na_non_potable)
    elif fill_na_method == "mean by class with noise":
        mask = data["Potability"] == 1
        fill_na_potable = data[mask].mean()
        potable_std = data[mask].std()
        fill_na_non_potable = data[~mask].mean()
        non_potable_std = data[~mask].std()

        missing_potable_line = data[mask][data[mask].isna().sum(axis=1).astype(bool)]
        for i in missing_potable_line.index:
            random_sample = pd.Series(
                np.random.normal(fill_na_potable, potable_std),
                index=fill_na_potable.index
            )
            data.loc[i] = data.loc[i].fillna(random_sample)
        missing_non_potable_line = data[~mask][data[~mask].isna().sum(axis=1).astype(bool)]
        for i in missing_non_potable_line.index:
            random_sample = pd.Series(
                np.random.normal(fill_na_non_potable, non_potable_std),
                index=fill_na_non_potable.index
            )
            data.loc[i] = data.loc[i].fillna(random_sample)
    elif fill_na_method == "median by class with noise":
        mask = data["Potability"] == 1
        fill_na_potable = data[mask].median()
        potable_std = data[mask].std()
        fill_na_non_potable = data[~mask].median()
        non_potable_std = data[~mask].std()

        missing_potable_line = data[mask][data[mask].isna().sum(axis=1).astype(bool)]
        for i in missing_potable_line.index:
            random_sample = pd.Series(
                np.random.normal(fill_na_potable, potable_std),
                index=fill_na_potable.index
            )
            data.loc[i] = data.loc[i].fillna(random_sample)
        missing_non_potable_line = data[~mask][data[~mask].isna().sum(axis=1).astype(bool)]
        for i in missing_non_potable_line.index:
            random_sample = pd.Series(
                np.random.normal(fill_na_non_potable, non_potable_std),
                index=fill_na_non_potable.index
            )
            data.loc[i] = data.loc[i].fillna(random_sample)
    elif fill_na_method == "mean":
        data.fillna(data.mean(), inplace=True)
    elif fill_na_method == "median":
        data.fillna(data.median(), inplace=True)
    elif fill_na_method == "dropna":
        data.dropna(axis=0, inplace=True)

    if remove_outliers:
        # ------ removing outliers ------ #
        d1 = data.quantile(0.01)
        d9 = data.quantile(0.99)
        data = data[~((data < d1) | (data > d9)).sum(axis=1).astype(bool)]

    x = data.drop("Potability", axis=1)
    y = data["Potability"]

    if normalize_features:
        x = (x - x.mean()) / x.std()

    if reduce_dimension == "PCA":
        # ------ PCA ------ #
        pca = PCA(n_components=3)
        x = pca.fit_transform(x)

    if model_used == "Random Forest":
        # ------ Random Forest ------ #
        model = RandomForestClassifier()
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(model, x, y, scoring="accuracy", cv=cv, n_jobs=1, error_score="raise")
    else:
        n_scores = None

    return n_scores.mean()
