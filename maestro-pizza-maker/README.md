# maestro-pizza-maker

## Guide for the coding exercise

### Introduction

This exercise is designed to test your ability to write code that is easy to understand, maintain, and test, as well as the ability to use the tools/methods that are commonly used in our community. The exercise is designed to be completed in 2-3 hours. If you are not able to complete it in this time, please do not worry. We are more interested in the quality of your code than the speed at which you write it.

You will be asked to follow the instructions in this `README.md` file and also detailed instructions in every file.

### Quick overview of the project

This project is supposed to help the Maestro Pizza Maker to be more efficient in his work. The project is divided into two parts: the `maestro_pizza_maker` package and the `notebooks` directory.
Some parts of the project are already implemented and some are not. Your task is to implement the missing parts. Please read the instructions carefully and everytime you see `TODO` in the code, implement the missing part.

There are two main classes in the `maestro_pizza_maker` package: `Pizza` and `PizzaMenu`. the `Pizza` class represents a single pizza and the `PizzaMenu` class represents a menu of pizzas. The `PizzaMenu` class is implemented in the `maestro_pizza_maker/pizza_menu.py` file and the `Pizza` class is implemented in the `maestro_pizza_maker/pizza.py` file. 

`Pizza` consists of predefined ingredients, which are represented by the `PizzaIngredient` class. The `PizzaIngredients` class is implemented in the `maestro_pizza_maker/ingredients.py` file. The `PizzaIngredients` class is already implemented and you do not need to change it. Here just please carefully go through the code and get familiar with it. Special atttention should be paid to the types of `PizzaIngredient` class attributes.

There is also `sand_box` directory, which contains some code that is not meant to be interactited with. It just provides drawings from Multivariate Normal Distribution used in the `PizzaIngredients` class.

Please go through the code and try to understand, how the code works. If it is not clear to you after first reading, please don't worry, the exercise is designed to guide you through the code step by step.

### How to proceed with the exercise

1. *Fork* this repository to your own GitHub account.
2. *Clone* the forked repository to your local machine.
3. Install the requirements from the `requirements.txt` file via pip or conda or from `pyproject.toml` via poetry.
4. Follow the instructions here in the `README.md` file and also detailed instructions in each file.
5. Solving the exercise in the order listed below is recommended.
6. *Commit* and *push* your changes to your forked repository at least one day before the interview.
7. *Send* us the link to your forked repository.

### Exercise

1. Implement `average_fat` property in the `Pizza` class (file `maestro_pizza_maker/pizza.py` line 78).
    1.1. Read TODO properly.
    1.2. Optional: Consider writing a test for this property to make sure it works properly and all the objects are created properly. It will help you to understand the code better.
2. Implement `name` property in the `Pizza` class (file `maestro_pizza_maker/pizza.py` line 92).
    2.1. Read TODO properly.
    2.2. Optional: Consider writing a test for this property.
3. Implement `taste` property in the `Pizza` class (file `maestro_pizza_maker/pizza.py` line 99).
    3.1. Read TODO properly.
    3.2. Optional: Consider writing a test for this property.
4. Implement `to_dataframe` method in the `PizzaMenu` class (file `pizza_maker/pizza_menu.py` line 15).
    4.1. Read TODO properly.
    4.2. Optional: Consider writing a test for this method.
5. Implement properites `cheapest_pizza` and`most_caloric_pizza` in the `PizzaMenu` class (file `maestro_pizza_maker/pizza_menu.py` line 39, 44).
    5.1. Optional: Consider writing a test for these properties.
    5.2. Optional: Consider writing other properties that might be useful.
6. Implement method `get_most_fat_pizza` in the `PizzaMenu` class (file `maestro_pizza_maker/pizza_menu.py` line 48).
    6.1. Read TODO properly.
    6.2. Optional: Consider writing a test for this method.
7. Implement `add_pizza` method in the `PizzaMenu` class (file `maestro_pizza_maker/pizza_menu.py` line 53).
    7.1. Optional: Consider writing a test for this method.
8. Implement `remove_pizza` method in the `PizzaMenu` class (file `maestro_pizza_maker/pizza_menu.py` line 57).
    8.1. Read TODO properly.
    8.2. Optional: Consider writing a test for this method.
9. Implement `__len__` method in the `PizzaMenu` class (file `maestro_pizza_maker/pizza_menu.py` line 63).
    9.1. Optional: Consider writing a unit test for this method using `pytest`.
10. Implement methods `menu_sensitivity_protein`, `menu_sensitivity_carbs`, `menu_sensitivity_fat` in the file `maestro_pizza_maker/pizza_sensitivities.py`.
    10.1. Read TODO properly.
11. Implement methods `taste_at_risk_pizza`, `taste_at_risk_menu`, `conditonal_taste_at_risk_pizza` and `conditonal_taste_at_risk_menu` in the file `maestro_pizza_maker/taste_at_risk.py`.
    11.1. Read TODO properly.
12. Design and implement optimization model `maximize_taste_penalty_price` in the file `maestro_pizza_maker/pizza_optimizer.py` (line 193).
    12.1. Read TODO properly.
    12.2 Find inspiration in the `minimize_price` model (file `maestro_pizza_maker/pizza_optimizer.py`, line 46)
13. Go to the notebook `notebooks/menu_analysis.ipynb` and follow the instructions there.
    13.1. Follow the instructions in the notebook cell by cell.
    13.2 Creativity is welcome.

Good luck!