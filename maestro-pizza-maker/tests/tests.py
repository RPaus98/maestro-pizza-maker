import unittest
from maestro_pizza_maker.pizza_menu import PizzaMenu
from maestro_pizza_maker.pizza import Pizza
from maestro_pizza_maker.ingredients import PizzaIngredients
from maestro_pizza_maker.pizza_sensitivities import *
from maestro_pizza_maker.taste_at_risk import *

class Test(unittest.TestCase):

    def test_avarage_fat_simple_pizza(self):
        test_pizza = Pizza(
            sauce=PizzaIngredients.CREAM_SAUCE,
            dough=PizzaIngredients.CLASSIC_DOUGH,
        )
        sauce=PizzaIngredients.CREAM_SAUCE
        dough=PizzaIngredients.CLASSIC_DOUGH
        test_avg = np.mean(np.array([sauce.value.fat, dough.value.fat]))
        self.assertEqual(test_pizza.average_fat, test_avg)


if __name__ == '__main__':
    unittest.main()