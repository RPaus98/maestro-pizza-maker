import unittest
import pandas as pd 
from maestro_pizza_maker.pizza_menu import PizzaMenu
from maestro_pizza_maker.pizza import Pizza
from maestro_pizza_maker.ingredients import PizzaIngredients
from maestro_pizza_maker.pizza_sensitivities import *
from maestro_pizza_maker.taste_at_risk import *

class Tests(unittest.TestCase):

    def test_avarage_fat_simple_pizza(self):
        test_pizza = Pizza(
            sauce=PizzaIngredients.CREAM_SAUCE,
            dough=PizzaIngredients.CLASSIC_DOUGH,
        )
        sauce=PizzaIngredients.CREAM_SAUCE
        dough=PizzaIngredients.CLASSIC_DOUGH
        test_avg = np.mean(np.array([sauce.value.fat, dough.value.fat]))
        self.assertEqual(test_pizza.average_fat, test_avg)
    
    def test_unique_name_pizza(self):
        names = []
        test = int(10e5)
        for _ in range(test):
            pizza = Pizza(
                        sauce=PizzaIngredients.CREAM_SAUCE,
                        dough=PizzaIngredients.CLASSIC_DOUGH
                    )
            names.append(pizza.name)
        df = pd.DataFrame()
        df["Names"] = names 
        self.assertEqual(len(df.Names.unique()), test)

if __name__ == '__main__':
    unittest.main()