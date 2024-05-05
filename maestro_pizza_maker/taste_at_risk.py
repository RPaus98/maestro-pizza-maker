# The maestro pizza maker is aware of the fact that the fat content of the ingredients is random and it is not always the same.
# Since fat is the most important factor in taste, the maestro pizza maker wants to know how risky his pizza menu is.

# TODO: define 2 risk measures for the pizza menu and implement them (1 - Taste at Risk (TaR), 2 - Conditional Taste at Risk (CTaR), also known as Expected Shorttaste (ES)

from maestro_pizza_maker.pizza import Pizza
from maestro_pizza_maker.pizza_menu import PizzaMenu
import numpy as np
from scipy.stats import multivariate_normal

def taste_at_risk_pizza(pizza: Pizza, quantile: float) -> float:
    # TODO: implement the taste at risk measure for a pizza
    # quantile is the quantile that we want to consider
    # Hint: Similarity between the Taste at Risk and the Value at Risk is not a coincidence or is it?
    # Hint: Use function taste from Pizza class, but be aware that the higher the taste, the better -> the lower the taste, the worse
    
    # We focus on the left tail of the taste distribution.
    if quantile>0.5: quantile = 1 - quantile
    
    return np.quantile(pizza.taste, q=quantile)

def taste_at_risk_menu(menu: PizzaMenu, quantile: float) -> float:
    # TODO: implement the taste at risk measure for a menu
    # quantile is the quantile that we want to consider
    # Hint: the taste of the whole menu is the sum of the taste of all pizzas in the menu, or? ;)

    # We focus on the left tail of the taste distribution.
    if quantile>0.5: quantile = 1 - quantile
    
    sum_taste: np.ndarray = sum(pizza.taste for pizza in menu.pizzas)
    return np.quantile(sum_taste, q=quantile)


def conditional_taste_at_risk_pizza(pizza: Pizza, quantile: float) -> float:
    # TODO: implement the conditional taste at risk measure for a pizza
    # quantile is the quantile that we want to consider
    # Hint: Simmilarity between the Conditional Taste at Risk and the Conditional Value at Risk is not a coincidence or is it?

    # We focus on the left tail of the taste distribution.
    if quantile>0.5: quantile = 1 - quantile

    TaR: float = taste_at_risk_pizza(pizza=pizza, quantile=quantile)
    taste: np.ndarray = pizza.taste
    return taste[taste <= TaR].mean()


def conditional_taste_at_risk_menu(menu: PizzaMenu, quantile: float) -> float:
    # TODO: implement the conditional taste at risk measure for a menu
    # Hint: the taste of the whole menu is the sum of the taste of all pizzas in the menu, or? ;) (same as for the taste at risk)

    # We focus on the left tail of the taste distribution.
    if quantile>0.5: quantile = 1 - quantile

    TaR: float = taste_at_risk_menu(menu=menu, quantile=quantile)
    taste: np.ndarray = sum(pizza.taste for pizza in menu.pizzas)
    return taste[taste <= TaR].mean()

