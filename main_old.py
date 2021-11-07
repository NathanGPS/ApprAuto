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
    seed = 2021
    fill_na_method = "mean by class"
    remove_outliers = True
    normalize_features = False
    reduce_dimension = None
    model_used = "Random Forest"
    data_augmentation = True


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
def main(fill_na_method, remove_outliers, normalize_features, reduce_dimension, model_used, seed, data_augmentation):
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
    elif fill_na_method == 'with zero':
        data.fillna(0, inplace=True)

    if remove_outliers:
        # ------ removing outliers ------ #
        d1 = data.quantile(0.1)
        d9 = data.quantile(0.9)
        data = data[~((data < d1) | (data > d9)).sum(axis=1).astype(bool)]
        print('data.shape ={}'.format(data.shape))
        
        # columns = [x for x in data.columns if x != 'Potability']
        # def treat_outliers(df,columns,target,label):
        #     for c in columns:
        #         q1 = np.quantile(df.loc[df[target] == label,c],0.25)
        #         q3 = np.quantile(df.loc[df[target] == label,c],0.75)
        #         df.loc[df[target] == label,c] = df.loc[df[target] == label,c].apply(lambda x: q1 if x<q1 else x)
        #         df.loc[df[target] == label,c] = df.loc[df[target] == label,c].apply(lambda x: q3 if x>q3 else x)
        # treat_outliers(data,columns,'Potability',1)

        print('data.shape ={}'.format(data.shape))

        

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
        params = {
            "min_samples_split": [10,20,100],
            "max_depth": [5,10,50],
            "min_samples_leaf": [10,20,50],
            "max_leaf_nodes": [10,20,100],
            "max_features": [9,5]
            }
        if data_augmentation:
            kf = KFold(n_splits=10, random_state=1, shuffle=True)
            acc_score = []
            for train_index, test_index in kf.split(x):
                model = RandomForestClassifier()
                print(type(train_index))
                x_train, x_test = x.iloc[list(train_index)], x.iloc[list(test_index)]
                y_train, y_test = y.iloc[list(train_index)], y.iloc[list(test_index)]
                # --- Data augmentation --- #
                x_train, y_train = SMOTE(random_state=1,n_jobs=-1).fit_resample(x_train,y_train)
                sc = StandardScaler()
                x_train[x_train.columns] = sc.fit_transform(x_train)
                x_test[x_test.columns] = sc.transform(x_test)
                model.fit(x_train,y_train)
                pred_values = model.predict(x_test)
                acc = accuracy_score(pred_values , y_test)
                acc_score.append(acc)
            n_scores = np.array(acc_score)

            # x_train, x_test, y_train, y_test = train_test_split(x, y ,train_size = 0.7,random_state = 1)
            # x_train, y_train = SMOTE(random_state=1,n_jobs=-1).fit_resample(x_train,y_train)
            # sc = StandardScaler()
            # x_train[x_train.columns] = sc.fit_transform(x_train)
            # x_test[x_test.columns] = sc.transform(x_test)
            # x_train = sm.add_constant(x_train)
            # x_test = sm.add_constant(x_test)
            # model.fit(x_train,y_train)
            # pred_values = model.predict(x_test)
            # acc = accuracy_score(pred_values , y_test)
            # n_scores = np.array([acc])

            # dt = DecisionTreeClassifier(random_state=1)
            # params = {
            #         "min_samples_split": [10,20,100],
            #         "max_depth": [5,10,50],
            #         "min_samples_leaf": [10,20,50],
            #         "max_leaf_nodes": [10,20,100]
            #     }

            # dt_grid = GridSearchCV(estimator=dt,param_grid=params,cv=5,
            # scoring='balanced_accuracy',verbose=0,n_jobs = -1).fit(x_train,y_train)
            # dt = dt_grid.best_estimator_
            # pred_values = dt.predict(x_test)

            # rf_grid = GridSearchCV(estimator=model,
            # param_grid=params,cv=5,scoring='balanced_accuracy',
            # verbose=10,n_jobs = -1).fit(x_train,y_train)
            # rf = rf_grid.best_estimator_
            # pred_values = rf.predict(x_test)
            # acc = accuracy_score(pred_values , y_test)
            # n_scores = np.array([acc])
            # cf = confusion_matrix(pred_values,y_test)

            # def metrics(cf):
            #     acc = (cf[0,0] + cf[1,1])/(cf[0,0]+ cf[0,1] + cf[1,0] + cf[1,1])
            #     recall = (cf[1,1])/(cf[1,0] + cf[1,1])
            #     specificty = (cf[0,0])/(cf[0,0] + cf[0,1])
            #     print("accuracy = {0}\n\nSensitivity(TPR) = {1}\n\nSpecificity = {2}".format(acc,recall,specificty))
                
            # metrics(cf)

        else:
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            n_scores = cross_val_score(model, x, y, scoring="accuracy", cv=cv, n_jobs=1, error_score="raise")
    else:
        n_scores = None
    print('n_scores = {}'.format(list(n_scores)))
    return n_scores.mean()
