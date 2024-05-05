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
    """
    Objective Function:
    \[
    \{minimize} \sum_{i=1}^{n} (x_i \cdot \{price}_i)
    \]

    Subject to constraints:
    \[
    &\sum_{i=1}^{n} (x_i \cdot \{protein}_i) \geq \{constraints\_values.protein.min} \\
    &\sum_{i=1}^{n} (x_i \cdot \{protein}_i) \leq \{constraints\_values.protein.max} \\
    &\{Similar constraints for fat, carbohydrates, and calories} \\
    &\sum_{i=1}^{n} x_i = \{constraints\_ingredients.dough} {(for dough)} \\
    &\sum_{i=1}^{n} x_i = \{constraints\_ingredients.sauce} {(for sauce)} \\
    &\sum_{i=1}^{n} x_i = \{constraints\_ingredients.cheese} {(for cheese)} \\
    &\sum_{i=1}^{n} x_i = \{constraints\_ingredients.meat} {(for meat)} \\
    &\sum_{i=1}^{n} x_i = \{constraints\_ingredients.vegetables} {(for vegetables)} \\
    &\sum_{i=1}^{n} x_i = \{constraints\_ingredients.fruits} {(for fruits)}
    \]

    Where:
    - \( x_i \) is a binary decision variable representing whether ingredient \( i \) is included in the pizza.
    - \( \{price}_i \), \( \{protein}_i \), etc., are properties of ingredient \( i \) (price, protein content, etc.).
    - \( \{constraints\_values.protein.min} \), \( \{constraints\_values.protein.max} \), etc., are the minimum and maximum constraints on nutritional values.
    - \( \{constraints\_ingredients.dough} \), etc., are the constraints on the number of ingredients of each type to include in the pizza.
    """
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
    
    """
    Objective Function:
    \[
    \{maximize} \left( \sum_{i=1}^{n} \left( x_i \cdot \E({taste}_i) \right) - \lambda \cdot \left( \sum_{i=1}^{n} \left( x_i \cdot \{price}_i \right) \right) \right)
    \]

    Subject to constraints:
    \[
    &\sum_{i=1}^{n} \left( x_i \cdot \{protein}_i \right) \geq \{constraints\_values.protein.min} \\
    &\sum_{i=1}^{n} \left( x_i \cdot \{protein}_i \right) \leq \{constraints\_values.protein.max} \\
    &\{Similar constraints for fat, carbohydrates, and calories} \\
    &\sum_{i=1}^{n} x_i = \{constraints\_ingredients.dough} {(for dough)} \\
    &\sum_{i=1}^{n} x_i = \{constraints\_ingredients.sauce} {(for sauce)} \\
    &\sum_{i=1}^{n} x_i = \{constraints\_ingredients.cheese} {(for cheese)} \\
    &\sum_{i=1}^{n} x_i = \{constraints\_ingredients.meat} {(for meat)} \\
    &\sum_{i=1}^{n} x_i = \{constraints\_ingredients.vegetables} {(for vegetables)} \\
    &\sum_{i=1}^{n} x_i = \{constraints\_ingredients.fruits} {(for fruits)}
    \]

    Where:
    - \( x_i \) is a binary decision variable representing whether ingredient \( i \) is included in the pizza.
    - \( \{taste}_i \) is the taste contribution of ingredient \( i \) based on its fat content.
    - \( \{price}_i \) is the price of ingredient \( i \).
    - \( \lambda \) is a parameter controlling the trade-off between taste and price (given as `lambda_param`).
    - \( \{constraints\_values} \) and \( \{constraints\_ingredients} \) represent the constraints on nutritional values and ingredient types, respectively.
    """

    # TODO: implement this function (description at the top of the file)
    # recomendation: use latex notation to describe the suggested model
    
    model = Model()

    # sets
    ingredients = [ingredient for ingredient in PizzaIngredients]
    ingredients_names = [ingredient.name for ingredient in ingredients]

    # variables: 1. Variable Setup: The sets up variables for each ingredient in the pizza.
    x = [
        model.add_var(var_type=INTEGER, lb=0, ub=1, name=ingredient)
        for ingredient in ingredients_names
    ]

    # 2.Taste Calculation: It calculates the taste of the pizza based on the fat content of each ingredient, 
    # considering different types of ingredients (dough, sauce, cheese, etc.).
    # REMAINDER: taste = 0.05 * fat_dough + 0.2 * fat_sauce + 0.3 * fat_cheese + 0.1 * fat_fruits + 0.3 * fat_meat + 0.05 * fat_vegetables
    # taste is linear combination of normally distributed fats
    taste = []
    for ingredient in ingredients:
        name = ingredient.value.type.value
        if name == "dough":
            taste.append(ingredient.value.fat * 0.05)
        elif name == "sauce":
            taste.append(ingredient.value.fat * 0.2) 
        elif name == "cheese":
            taste.append(ingredient.value.fat * 0.3)
        elif name == "fruit":
            taste.append(ingredient.value.fat * 0.1)
        elif name == "meat":
            taste.append(ingredient.value.fat * 0.05)
        elif name == "vegetable":
            taste.append(ingredient.value.fat * 0.05)

    # 3. Objective set-up: The objective function aims to maximize the taste of the pizza
    # minus a penalty term for its price, controlled by the parameter lambda_param (given from Maestro Pizza).
    expected_tastes = np.average(taste, axis=1)
    mean_pizza_taste = xsum(x[i] * expected_tastes[i] for i in range(len(expected_tastes)))
    
    #price as previous function
    price = xsum(x[i] * ingredients[i].value.price for i in range(len(ingredients)))

    # Objective function:
    model.objective = maximize(mean_pizza_taste - lambda_param * price)

    # 4. Constraint Setup: Constraints are added for various nutritional values (protein, fat, carbohydrates, calories) 
    # and the quantities of different types of ingredients (dough, sauce, cheese, etc.).
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

    # 5. Optimization of the Model
    model.optimize()

    # check solution
    if model.status != OptimizationStatus.OPTIMAL:
        raise Exception(
            "The model is not optimal -> likely no solution found (infeasible))"
        )

    # 6. Solution Extraction: it returns a Pizza object containing the selected ingredients.
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
