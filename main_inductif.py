from re import X
import pandas as pd
from sacred import Experiment
from data import get_data, data_ingredient
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from mpl_toolkits import mplot3d
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ex = Experiment("drinkable water", ingredients=[data_ingredient])


@ex.config
def config():
    # seed = 2021
    seed = 56
    fill_na_method = "mean by class"
    treat_outliers = dict(used = True, version = 'put quantile by classes')
    normalize_features = False
    reduce_dimension = None
    model_used = "Random Forest"
    data_augmentation = True
    k_folds = dict(used = True, nbr_fold = 10)


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
def main(fill_na_method, treat_outliers, normalize_features, 
         reduce_dimension, model_used, seed, data_augmentation,
         k_folds):
    # --- Setting the seed --- #
    np.random.seed(seed)

    data = get_data()

    acc_list = []

    if not k_folds['used']:
        k_folds['nbr_fold'] = 1
    kf = KFold(n_splits=k_folds['nbr_fold'], random_state=1, shuffle=True)

    x = data.drop("Potability", axis=1)
    y = data["Potability"] 

    for train_index, test_index in kf.split(x):

        x_train, x_test = x.iloc[list(train_index)], x.iloc[list(test_index)]
        y_train, y_test = y.iloc[list(train_index)], y.iloc[list(test_index)]
        print(y_train)

        if fill_na_method == "mean by class":
            mask = y_train == 1
            fill_na_potable = x_train[mask].mean()
            fill_na_non_potable = x_train[~mask].mean()

            x_train[mask] = x_train[mask].fillna(fill_na_potable)
            x_train[~mask] = x_train[~mask].fillna(fill_na_non_potable)
            
            mask_test = y_test == 1
            x_test[mask_test] = x_test[mask_test].fillna(fill_na_potable)
            x_test[~mask_test] = x_test[~mask_test].fillna(fill_na_non_potable)

        elif fill_na_method == "median by class":
            mask = y_train== 1
            fill_na_potable = x_train[mask].median()
            fill_na_non_potable = x_train[~mask].median()

            x_train[mask] = x_train[mask].fillna(fill_na_potable)
            x_train[~mask] = x_train[~mask].fillna(fill_na_non_potable)
            
            mask_test = y_test == 1
            x_test[mask_test] = x_test[mask_test].fillna(fill_na_potable)
            x_test[~mask_test] = x_test[~mask_test].fillna(fill_na_non_potable)

        elif fill_na_method == "mean by class with noise":
            mask = y_train == 1
            fill_na_potable = x_train[mask].mean()
            potable_std = x_train[mask].std()
            fill_na_non_potable = x_train[~mask].mean()
            non_potable_std = x_train[~mask].std()

            missing_potable_line = x_train[mask][x_train[mask].isna().sum(axis=1).astype(bool)]
            for i in missing_potable_line.index:
                random_sample = pd.Series(
                    np.random.normal(fill_na_potable, potable_std),
                    index=fill_na_potable.index
                )
                x_train.loc[i] = x_train.loc[i].fillna(random_sample)
            missing_non_potable_line = x_train[~mask][x_train[~mask].isna().sum(axis=1).astype(bool)]
            for i in missing_non_potable_line.index:
                random_sample = pd.Series(
                    np.random.normal(fill_na_non_potable, non_potable_std),
                    index=fill_na_non_potable.index
                )
                x_train.loc[i] = x_train.loc[i].fillna(random_sample)

            mask_test = y_test == 1

            missing_potable_line = x_test[mask_test][x_test[mask_test].isna().sum(axis=1).astype(bool)]
            for i in missing_potable_line.index:
                random_sample = pd.Series(
                    np.random.normal(fill_na_potable, potable_std),
                    index=fill_na_potable.index
                )
                x_test.loc[i] = x_test.loc[i].fillna(random_sample)
            missing_non_potable_line = x_test[~mask_test][x_test[~mask_test].isna().sum(axis=1).astype(bool)]
            for i in missing_non_potable_line.index:
                random_sample = pd.Series(
                    np.random.normal(fill_na_non_potable, non_potable_std),
                    index=fill_na_non_potable.index
                )
                x_test.loc[i] = x_test.loc[i].fillna(random_sample)

        elif fill_na_method == "median by class with noise":
            mask = y_train == 1
            fill_na_potable = x_train[mask].median()
            potable_std = x_train[mask].std()
            fill_na_non_potable = x_train[~mask].median()
            non_potable_std = x_train[~mask].std()

            missing_potable_line = x_train[mask][x_train[mask].isna().sum(axis=1).astype(bool)]
            for i in missing_potable_line.index:
                random_sample = pd.Series(
                    np.random.normal(fill_na_potable, potable_std),
                    index=fill_na_potable.index
                )
                x_train.loc[i] = x_train.loc[i].fillna(random_sample)
            missing_non_potable_line = x_train[~mask][x_train[~mask].isna().sum(axis=1).astype(bool)]
            for i in missing_non_potable_line.index:
                random_sample = pd.Series(
                    np.random.normal(fill_na_non_potable, non_potable_std),
                    index=fill_na_non_potable.index
                )
                x_train.loc[i] = x_train.loc[i].fillna(random_sample)

            mask_test = y_test == 1

            missing_potable_line = x_test[mask_test][x_test[mask_test].isna().sum(axis=1).astype(bool)]
            for i in missing_potable_line.index:
                random_sample = pd.Series(
                    np.random.normal(fill_na_potable, potable_std),
                    index=fill_na_potable.index
                )
                x_test.loc[i] = x_test.loc[i].fillna(random_sample)
            missing_non_potable_line = x_test[~mask_test][x_test[~mask_test].isna().sum(axis=1).astype(bool)]
            for i in missing_non_potable_line.index:
                random_sample = pd.Series(
                    np.random.normal(fill_na_non_potable, non_potable_std),
                    index=fill_na_non_potable.index
                )
                x_test.loc[i] = x_test.loc[i].fillna(random_sample)

        elif fill_na_method == "mean":
            x_train.fillna(x_train.mean(), inplace=True)
            x_test.fillna(x_train.mean(), inplace=True)

        elif fill_na_method == "median":
            x_train.fillna(x_train.median(), inplace=True)
            x_test.fillna(x_train.median(), inplace=True)

        elif fill_na_method == "dropna":
            x_train.dropna(axis=0, inplace=True)
            
            x_test.dropna(axis=0, inplace=True)
            
        elif fill_na_method == 'with zero':
            x_train.fillna(0, inplace=True)
            x_test.fillna(0, inplace=True)

        if treat_outliers['used']:

            # ------ removing outliers ------ #
            if treat_outliers['version'] == 'remove':
                d1 = x_train.quantile(0.1)
                d9 = x_train.quantile(0.9)
                x_train = x_train[~((x_train < d1) | (x_train > d9)).sum(axis=1).astype(bool)]
                x_test = x_test[~((x_test < d1) | (x_test > d9)).sum(axis=1).astype(bool)]
            elif treat_outliers['version'] == 'put quantile only potability':
                y_train = y_train[x_train.index]
                y_test = y_test[x_test.index]
                for c in x_train.columns:
                    q1 = np.quantile(x_train.loc[y_train   == 1,c],0.25)
                    q3 = np.quantile(x_train.loc[y_train   == 1,c],0.75)
                    x_train.loc[y_train==1 ,c] = x_train.loc[y_train ==1 ,c].apply(lambda x: q1 if x<q1 else x)
                    x_train.loc[y_train==1,c] = x_train.loc[y_train  ==1,c].apply(lambda x: q3 if x>q3 else x)

                    x_test.loc[y_test==1 ,c] = x_test.loc[y_test==1 ,c].apply(lambda x: q1 if x<q1 else x)
                    x_test.loc[y_test==1,c] = x_test.loc[y_test==1 ,c].apply(lambda x: q3 if x>q3 else x)
            elif treat_outliers['version'] == 'put quantile by classes':
                y_train = y_train[x_train.index]
                y_test = y_test[x_test.index]
                for c in x_train.columns:
                    for potability in [0 , 1]:
                        q1 = np.quantile(x_train.loc[y_train   == potability,c],0.25)
                        q3 = np.quantile(x_train.loc[y_train   == potability,c],0.75)
                        x_train.loc[y_train==potability ,c] = x_train.loc[y_train==potability,c].apply(lambda x: q1 if x<q1 else x)
                        x_train.loc[y_train==potability,c] = x_train.loc[y_train==potability,c].apply(lambda x: q3 if x>q3 else x)

                        x_test.loc[y_test==potability ,c] = x_test.loc[y_test==potability,c].apply(lambda x: q1 if x<q1 else x)
                        x_test.loc[y_test==potability,c] = x_test.loc[y_test==potability,c].apply(lambda x: q3 if x>q3 else x)
 
        y_train = y_train[x_train.index]
        y_test = y_test[x_test.index]

        if data_augmentation:
            x_train, y_train = SMOTE(random_state=1,n_jobs=-1).fit_resample(x_train,y_train)

        if normalize_features:
            sc = StandardScaler()
            x_train[x_train.columns] = sc.fit_transform(x_train)
            x_test[x_test.columns] = sc.transform(x_test)

        if reduce_dimension == "PCA":
            # ------ PCA ------ #
            pca = PCA(n_components=3)
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)

        if model_used == "Random Forest":
            # ------ Random Forest ------ #
            model = RandomForestClassifier()
            # --- Data augmentation --- #
                
            model.fit(x_train,y_train)
            pred_values = model.predict(x_test)
            acc = accuracy_score(pred_values , y_test)
            acc_list.append(acc)
            
        else:
            acc_list = [None]
            break
    n_scores = np.array(acc_list)
    print('n_scores = {}'.format(list(n_scores)))
    return n_scores.mean()
