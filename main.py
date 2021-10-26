import pandas as pd
from sacred import Experiment
from data import get_data, data_ingredient

ex = Experiment("drinkable water", ingredients=[data_ingredient])


@ex.config
def config():
    pass


@ex.automain
def main():
    data = get_data()

    return 0
