# class representing a pizza ingredient

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Literal, Union
import pandas as pd
from maestro_pizza_maker.sand_box.fat_generator import FAT_SIMULATIONS

# from numpy.random import normal, exponential, gamma, uniform
import numpy as np


class IngredientType(Enum):
    MEAT = "meat"
    VEGETABLE = "vegetable"
    CHEESE = "cheese"
    SAUCE = "sauce"
    DOUGH = "dough"
    FRUIT = "fruit"


@dataclass
class PizzaIngredient:
    name: str
    price: float
    type: IngredientType
    protein: float
    fat: np.array
    carbohydrates: float
    calories: float


# enum representing pizza ingredients


class PizzaIngredients(Enum):
    TOMATO_SAUCE = PizzaIngredient(
        name="TOMATO SAUCE",
        price=0.5,
        type=IngredientType.SAUCE,
        protein=0.5,
        fat=FAT_SIMULATIONS[0],
        carbohydrates=3.0,
        calories=20.0,
    )
    CREAM_SAUCE = PizzaIngredient(
        name="CREAM SAUCE",
        price=0.6,
        type=IngredientType.SAUCE,
        protein=0.6,
        fat=FAT_SIMULATIONS[1],
        carbohydrates=4.0,
        calories=30.0,
    )
    MOZZARELA = PizzaIngredient(
        name="MOZZRELA",
        price=1.0,
        type=IngredientType.CHEESE,
        protein=10.0,
        fat=FAT_SIMULATIONS[2],
        carbohydrates=0.0,
        calories=400.0,
    )
    CHEDDAR = PizzaIngredient(
        name="CHEDDAR",
        price=1.0,
        type=IngredientType.CHEESE,
        protein=10.0,
        fat=FAT_SIMULATIONS[3],
        carbohydrates=0.0,
        calories=400.0,
    )
    PARMESAN = PizzaIngredient(
        name="PARMESAN",
        price=1.0,
        type=IngredientType.CHEESE,
        protein=10.0,
        fat=FAT_SIMULATIONS[4],
        carbohydrates=0.0,
        calories=400.0,
    )
    BACON = PizzaIngredient(
        name="BACON",
        price=1.0,
        type=IngredientType.MEAT,
        protein=10.0,
        fat=FAT_SIMULATIONS[5],
        carbohydrates=0.0,
        calories=400.0,
    )
    SAUSAGE = PizzaIngredient(
        name="SAUSAGE",
        price=1.0,
        type=IngredientType.MEAT,
        protein=10.0,
        fat=FAT_SIMULATIONS[6],
        carbohydrates=0.0,
        calories=400.0,
    )
    HAM = PizzaIngredient(
        name="HAM",
        price=2.0,
        type=IngredientType.MEAT,
        protein=20.0,
        fat=FAT_SIMULATIONS[7],
        carbohydrates=0.0,
        calories=800.0,
    )
    MUSHROOMS = PizzaIngredient(
        name="MUSHROOMS",
        price=1.0,
        type=IngredientType.VEGETABLE,
        protein=5.0,
        fat=FAT_SIMULATIONS[8],
        carbohydrates=5.0,
        calories=50.0,
    )
    ONIONS = PizzaIngredient(
        name="ONIONS",
        price=1.0,
        type=IngredientType.VEGETABLE,
        protein=5.0,
        fat=FAT_SIMULATIONS[9],
        carbohydrates=5.0,
        calories=50.0,
    )
    PEPPER = PizzaIngredient(
        name="PEPPER",
        price=1.0,
        type=IngredientType.VEGETABLE,
        protein=5.0,
        fat=FAT_SIMULATIONS[10],
        carbohydrates=5.0,
        calories=50.0,
    )
    PINEAPPLE = PizzaIngredient(
        name="PINEAPPLE",
        price=1.0,
        type=IngredientType.FRUIT,
        protein=5.0,
        fat=FAT_SIMULATIONS[11],
        carbohydrates=5.0,
        calories=50.0,
    )
    APPLE = PizzaIngredient(
        name="APPLE",
        price=1.0,
        type=IngredientType.FRUIT,
        protein=5.0,
        fat=FAT_SIMULATIONS[12],
        carbohydrates=5.0,
        calories=50.0,
    )
    CLASSIC_DOUGH = PizzaIngredient(
        name="CLASSIC DOUGH",
        price=1.0,
        type=IngredientType.DOUGH,
        protein=10.0,
        fat=FAT_SIMULATIONS[13],
        carbohydrates=10.0,
        calories=100.0,
    )
    THIN_DOUGH = PizzaIngredient(
        name="THIN DOUGH",
        price=1.0,
        type=IngredientType.DOUGH,
        protein=10.0,
        fat=FAT_SIMULATIONS[14],
        carbohydrates=10.0,
        calories=100.0,
    )
    WHOLEMEAL_DOUGH = PizzaIngredient(
        name="WHOLEMEAL DOUGH",
        price=1.0,
        type=IngredientType.DOUGH,
        protein=10.0,
        fat=FAT_SIMULATIONS[15],
        carbohydrates=10.0,
        calories=100.0,
    )

    # create a dataframes with all ingredients
    @staticmethod
    def get_ingredients_df():
        ingredients = []
        for ingredient in PizzaIngredients:
            ingredients.append(
                {
                    "name": ingredient.value.name,
                    "price ": ingredient.value.price,
                    "type": ingredient.value.type,
                    "protein ": ingredient.value.protein,
                    "fat ": ingredient.value.fat,
                    "carbohydrates ": ingredient.value.carbohydrates,
                    "calories ": ingredient.value.calories,
                }
            )
        return pd.DataFrame(ingredients)
