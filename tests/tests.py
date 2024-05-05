import unittest
import pandas as pd 
from maestro_pizza_maker.pizza_menu import PizzaMenu
from maestro_pizza_maker.pizza import Pizza
from maestro_pizza_maker.ingredients import PizzaIngredients
from maestro_pizza_maker.pizza_sensitivities import *
from maestro_pizza_maker.taste_at_risk import *

class Tests(unittest.TestCase):

    test_pizza = Pizza(
            sauce=PizzaIngredients.CREAM_SAUCE,
            dough=PizzaIngredients.CLASSIC_DOUGH,
        )
    
    test_menu = pizza_menu = PizzaMenu(
        pizzas=[
            Pizza(
                sauce=PizzaIngredients.CREAM_SAUCE,
                dough=PizzaIngredients.CLASSIC_DOUGH,
                cheese=[PizzaIngredients.MOZZARELA],
                fruits=[PizzaIngredients.PINEAPPLE],
                meat=[PizzaIngredients.BACON, PizzaIngredients.HAM],
                vegetables=[PizzaIngredients.ONIONS],
            ),
            Pizza(
                sauce=PizzaIngredients.TOMATO_SAUCE,
                dough=PizzaIngredients.THIN_DOUGH,
                cheese=[PizzaIngredients.CHEDDAR],
                fruits=[PizzaIngredients.APPLE],
                meat=[PizzaIngredients.SAUSAGE],
                vegetables=[PizzaIngredients.MUSHROOMS, PizzaIngredients.PEPPER],
            ),
            Pizza(
                sauce=PizzaIngredients.CREAM_SAUCE,
                dough=PizzaIngredients.WHOLEMEAL_DOUGH,
                cheese=[PizzaIngredients.MOZZARELA, PizzaIngredients.CHEDDAR],
                fruits=[PizzaIngredients.PINEAPPLE, PizzaIngredients.APPLE],
                meat=[
                    PizzaIngredients.BACON,
                    PizzaIngredients.HAM,
                    PizzaIngredients.SAUSAGE,
                ],
                vegetables=[
                    PizzaIngredients.MUSHROOMS,
                    PizzaIngredients.ONIONS,
                    PizzaIngredients.PEPPER,
                ],
            ),
            Pizza(
                sauce=PizzaIngredients.TOMATO_SAUCE,
                dough=PizzaIngredients.CLASSIC_DOUGH,
                cheese=[PizzaIngredients.MOZZARELA],
                fruits=[PizzaIngredients.PINEAPPLE],
                meat=[PizzaIngredients.BACON, PizzaIngredients.HAM],
                vegetables=[PizzaIngredients.ONIONS],
            )
        ]
    )
    
    def test_avarage_fat_simple_pizza(self):
        sauce=PizzaIngredients.CREAM_SAUCE
        dough=PizzaIngredients.CLASSIC_DOUGH
        test_avg = np.mean(np.array([sauce.value.fat, dough.value.fat]))
        self.assertEqual(self.test_pizza.average_fat, test_avg)
    
    def test_unique_name_pizza(self):
        names = []
        test = int(10e5)
        for _ in range(test):
            names.append(self.test_pizza.name)
        df = pd.DataFrame()
        df["Names"] = names 
        self.assertEqual(len(df.Names.unique()), test)

    def test_TaR_symmetry(self):
        q1 = 0.1
        q2 = 0.9
        TaR_1 = taste_at_risk_pizza(pizza=self.test_pizza, quantile=q1)
        TaR_2 = taste_at_risk_pizza(pizza=self.test_pizza, quantile=q2)
        self.assertEqual(TaR_1, TaR_2)

    def test_most_caloric_pizza(self):
        df_test = self.pizza_menu.to_dataframe("calories", descendent=True)
        test_most_caloric_pizza = df_test.iloc[0]["calories"]
        most_caloric_pizza = pizza_menu.most_caloric_pizza
        self.assertEqual(test_most_caloric_pizza, most_caloric_pizza.calories)
    
    def test_cheapest_pizza(self):
        df_test = self.pizza_menu.to_dataframe("price", descendent=False)
        test_cheapest_pizza = df_test.iloc[0]["price"]
        cheapest_pizza = pizza_menu.cheapest_pizza
        self.assertEqual(test_cheapest_pizza, cheapest_pizza.price)
    
    def test_len(self):
        len_df = 4
        df_pizzas = self.test_menu.to_dataframe("price", descendent=False)
        self.assertEqual(len_df, df_pizzas.__len__())
    
    def test_add_pizza(self):
        test_menu = self.test_menu
        test_menu.add_pizza(self.test_pizza)
        self.assertEqual(len(test_menu), 5)
        test_menu.remove_pizza(self.test_pizza)

    def test_remove_pizza(self):
        test_menu = self.test_menu
        pizza_to_remove = test_menu.cheapest_pizza
        test_menu.remove_pizza(pizza_to_remove)
        self.assertEqual(len(test_menu), 3)
        test_menu.add_pizza(pizza_to_remove)

if __name__ == '__main__':
    unittest.main()