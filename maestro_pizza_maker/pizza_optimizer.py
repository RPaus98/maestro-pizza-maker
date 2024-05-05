# The maestro pizza maker is aware that pizza creation is a complex process and it is not always possible to create a pizza that satisfies all the constraints.
# Therefore maestro wants to create a pizza that will satisfy as many constraints as possible to avoid the risk creating disappointing pizzas like pizza Hawaii.
# To do so, maestro wants to use the optimization techniques.

# TODO: implement the pizza optimizer that will create a pizza that will satisfy all the constraints and will maximize following objective functions:
# obj = Expected_value(taste(pizza)) - lambda * price(pizza), where lambda is a parameter that will be provided by the maestro pizza maker
# hint: use the mip library and the following documentation: https://www.python-mip.com/
# hint: you can find inspiration in the minimize_price function


from dataclasses import dataclass

import numpy as np

from mip import Model, xsum, minimize, maximize,INTEGER, OptimizationStatus

from maestro_pizza_maker.ingredients import IngredientType, PizzaIngredients
from maestro_pizza_maker.pizza import Pizza


@dataclass
class ValueBounds:
    min: float = 0.0
    max: float = np.inf


@dataclass
class PizzaConstraintsValues:
    price: ValueBounds = ValueBounds()
    protein: ValueBounds = ValueBounds()
    fat: ValueBounds = ValueBounds()
    carbohydrates: ValueBounds = ValueBounds()
    calories: ValueBounds = ValueBounds()


@dataclass
class PizzaConstraintsIngredients:
    cheese: int = 0
    fruits: int = 0
    meat: int = 0
    vegetables: int = 0
    dough: int = 1
    sauce: int = 1


def minimize_price(
    constraints_values: PizzaConstraintsValues,
    constraints_ingredients: PizzaConstraintsIngredients,
) -> Pizza:
    """"""
    model = Model()

    # sets
    ingredients = [ingredient for ingredient in PizzaIngredients]
    ingredients_names = [ingredient.name for ingredient in ingredients]

    # variables
    x = [
        model.add_var(var_type=INTEGER, lb=0, ub=1, name=ingredient)
        for ingredient in ingredients_names
    ]

    # objective function
    model.objective = minimize(
        xsum(x[i] * ingredients[i].value.price for i in range(len(ingredients)))
    )

    # constraints
    model += (
        xsum(x[i] * ingredients[i].value.protein for i in range(len(ingredients)))
        >= constraints_values.protein.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.protein for i in range(len(ingredients)))
        <= constraints_values.protein.max
    )
    model += (
        xsum(x[i] * ingredients[i].value.fat.mean() for i in range(len(ingredients)))
        >= constraints_values.fat.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.fat.mean() for i in range(len(ingredients)))
        <= constraints_values.fat.max
    )
    model += (
        xsum(x[i] * ingredients[i].value.carbohydrates for i in range(len(ingredients)))
        >= constraints_values.carbohydrates.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.carbohydrates for i in range(len(ingredients)))
        <= constraints_values.carbohydrates.max
    )
    model += (
        xsum(x[i] * ingredients[i].value.calories for i in range(len(ingredients)))
        >= constraints_values.calories.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.calories for i in range(len(ingredients)))
        <= constraints_values.calories.max
    )

    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.DOUGH
        )
        == constraints_ingredients.dough
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.SAUCE
        )
        == constraints_ingredients.sauce
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.CHEESE
        )
        == constraints_ingredients.cheese
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.MEAT
        )
        == constraints_ingredients.meat
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.VEGETABLE
        )
        == constraints_ingredients.vegetables
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.FRUIT
        )
        == constraints_ingredients.fruits
    )

    # optimize
    model.optimize()

    # check solution
    if model.status != OptimizationStatus.OPTIMAL:
        raise Exception(
            "The model is not optimal -> likely no solution found (infeasible))"
        )

    # solution
    return Pizza(
        dough=[
            ingredients[i].value
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.DOUGH and x[i].x == 1
        ][0],
        sauce=[
            ingredients[i].value
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.SAUCE and x[i].x == 1
        ],
        cheese=[
            ingredients[i].value
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.CHEESE and x[i].x == 1
        ],
        meat=[
            ingredients[i].value
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.MEAT and x[i].x == 1
        ],
        vegetables=[
            ingredients[i].value
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.VEGETABLE and x[i].x == 1
        ],
        fruits=[
            ingredients[i].value
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.FRUIT and x[i].x == 1
        ],
    )


def maximize_taste_penalty_price(
    constraints_values: PizzaConstraintsValues,
    constraints_ingredients: PizzaConstraintsIngredients,
    lambda_param: float = 0.5,
) -> Pizza:
    # TODO: implement this function (description at the top of the file)
    # recomendation: use latex notation to describe the suggested model
    
    model = Model()

    # sets
    ingredients = [ingredient for ingredient in PizzaIngredients]
    ingredients_names = [ingredient.name for ingredient in ingredients]

    # variables
    x = [
        model.add_var(var_type=INTEGER, lb=0, ub=1, name=ingredient)
        for ingredient in ingredients_names
    ]

    # set up for pizza taste
    # REMAINDER: taste = 0.05 * fat_dough + 0.2 * fat_sauce + 0.3 * fat_cheese + 0.1 * fat_fruits + 0.3 * fat_meat + 0.05 * fat_vegetables
    taste = []
    for ingredient in ingredients:
        n = ingredient.value.type.value
        if n == "dough":
            taste.append(ingredient.value.fat * 0.05)
        elif n == "sauce":
            taste.append(ingredient.value.fat * 0.2) 
        elif n == "chees":
            taste.append(ingredient.value.fat * 0.3)
        elif n == "fruit":
            taste.append(ingredient.value.fat * 0.1)
        elif n == "meat":
            taste.append(ingredient.value.fat * 0.05)
        elif n == "vegetable":
            taste.append(ingredient.value.fat * 0.05)

    # expected value of normally distributed is the mean. Normally distributed times 
    # the fat vector constant is still normally distributed.
    mean_tastes = [fat.mean() for fat in taste]

    # objective setup
    # taste is linear combination of normally distributed fats
    mean_pizza_taste = xsum(x[i] * mean_tastes[i] for i in range(len(mean_tastes)))
    #price as previous function
    price = xsum(x[i] * ingredients[i].value.price for i in range(len(ingredients)))

    # objective function
    model.objective = maximize(mean_pizza_taste - lambda_param * price)

    # constraints
    model += (
        xsum(x[i] * ingredients[i].value.protein for i in range(len(ingredients)))
        >= constraints_values.protein.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.protein for i in range(len(ingredients)))
        <= constraints_values.protein.max
    )
    model += (
        xsum(x[i] * ingredients[i].value.fat.mean() for i in range(len(ingredients)))
        >= constraints_values.fat.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.fat.mean() for i in range(len(ingredients)))
        <= constraints_values.fat.max
    )
    model += (
        xsum(x[i] * ingredients[i].value.carbohydrates for i in range(len(ingredients)))
        >= constraints_values.carbohydrates.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.carbohydrates for i in range(len(ingredients)))
        <= constraints_values.carbohydrates.max
    )
    model += (
        xsum(x[i] * ingredients[i].value.calories for i in range(len(ingredients)))
        >= constraints_values.calories.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.calories for i in range(len(ingredients)))
        <= constraints_values.calories.max
    )

    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.DOUGH
        )
        == constraints_ingredients.dough
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.SAUCE
        )
        == constraints_ingredients.sauce
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.CHEESE
        )
        == constraints_ingredients.cheese
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.MEAT
        )
        == constraints_ingredients.meat
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.VEGETABLE
        )
        == constraints_ingredients.vegetables
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.FRUIT
        )
        == constraints_ingredients.fruits
    )

    # optimize
    model.optimize()

    # check solution
    if model.status != OptimizationStatus.OPTIMAL:
        raise Exception(
            "The model is not optimal -> likely no solution found (infeasible))"
        )

    # solution
    return Pizza(
        dough=[
            ingredients[i] 
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.DOUGH and x[i].x == 1
        ][0],
        sauce=[
            ingredients[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.SAUCE and x[i].x == 1
        ][0], 
        cheese=[
            ingredients[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.CHEESE and x[i].x == 1
        ],
        meat=[
            ingredients[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.MEAT and x[i].x == 1
        ],
        vegetables=[
            ingredients[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.VEGETABLE and x[i].x == 1
        ],
        fruits=[
            ingredients[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.FRUIT and x[i].x == 1
        ],
    )
