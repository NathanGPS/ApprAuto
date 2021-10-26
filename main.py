import pandas as pd
from sacred import Experiment
from data import get_data, data_ingredient
import matplotlib.pyplot as plt

ex = Experiment("drinkable water", ingredients=[data_ingredient])


@ex.config
def config():
    pass


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
def main():
    data = get_data()

    # Study missing data within features
    # missing_values = data.isna().sum() / len(data)
    # missing_values.sort_values(inplace=True, ascending=False)

    # plot_bar_charts(missing_values, title="Percentage of missing value per features")
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    # plot_pie_chart(data, "Sulfate", axes[0, 0], only_keep_na=True)
    # plot_pie_chart(data, "ph", axes[0, 1], only_keep_na=True)
    # plot_pie_chart(data, "Trihalomethanes", axes[1, 0], only_keep_na=True)
    # plot_pie_chart(data, "", axes[1, 1])
    # fig.suptitle(
    #     "Impact of missing features on potability",
    #     x=.5,
    #     y=0,
    #     va="bottom"
    # )
    # plt.show()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
    missing_sulfate = data[data["Sulfate"].isna()]
    missing_sulfate = missing_sulfate[["ph", "Trihalomethanes"]]
    missing_sulfate = missing_sulfate.isna().sum() / len(missing_sulfate)
    plot_bar_charts(missing_sulfate, title="Missing values within missing Sulfate", ax=axes[0])
    axes[0].tick_params(axis='x', rotation=0)

    missing_sulfate = data[data["ph"].isna()]
    missing_sulfate = missing_sulfate[["Sulfate", "Trihalomethanes"]]
    missing_sulfate = missing_sulfate.isna().sum() / len(missing_sulfate)
    plot_bar_charts(missing_sulfate, title="Missing values within missing pH", ax=axes[1])
    axes[1].tick_params(axis='x', rotation=0)

    missing_sulfate = data[data["Trihalomethanes"].isna()]
    missing_sulfate = missing_sulfate[["Sulfate", "ph"]]
    missing_sulfate = missing_sulfate.isna().sum() / len(missing_sulfate)
    plot_bar_charts(missing_sulfate, title="Missing values within missing Trihalomethanes",
                    ax=axes[2])
    axes[2].tick_params(axis='x', rotation=0)

    plt.show()

    return 0
