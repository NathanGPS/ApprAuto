from sacred import Ingredient
import pandas as pd

data_ingredient = Ingredient("data")


@data_ingredient.config
def config():
    data_path = "./data/drinking_water_potability.csv"


@data_ingredient.capture
def get_data(data_path):
    data = pd.read_csv(data_path)
    return data
