import pandas as pd
import random
from typing import List
from collections import Counter
from itertools import tee
from pulp import LpMinimize, LpProblem, LpVariable
from pulp.apis import PULP_CBC_CMD


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def select_random_transformers_set(transponders_list: List[str]):
    random_choice = Counter(
        [random.SystemRandom().choice(transponders_list) for _ in range(random.SystemRandom().randint(2, 4))])
    key_order = ['10G', '40G', '100G']
    return {k: random_choice[k] for k in key_order if k in random_choice}  # sort keys in transponders dict


def find_cheapest_set_of_transponders_for_edge(demand: float, transponders_cost: dict):
    # Create the model
    model = LpProblem(name="transponder-min-cost-problem", sense=LpMinimize)

    # Initialize the decision variables
    x = LpVariable(name="10G", lowBound=0, cat="Integer")
    y = LpVariable(name="40G", lowBound=0, cat="Integer")
    z = LpVariable(name="100G", lowBound=0, cat="Integer")

    # Add the constraints to the model
    model += (10 * x + 40 * y + 100 * z >= demand, "satisfy_demand_constraint")

    # Add the objective function to the model
    obj_func = transponders_cost['10G'] * x + transponders_cost['40G'] * y + transponders_cost['100G'] * z
    model += obj_func

    # Solve the problem
    model.solve(PULP_CBC_CMD(msg=False))
    total_cost = model.objective.value()
    transponders_set = {}
    for var in model.variables():
        transponders_set[var.name] = int(var.value())

    # print('\ndemand', demand)
    # print(f"total transponders cost: {total_cost}")
    # print('transponders_set', transponders_set)
    return transponders_set, total_cost


class Chromosome:
    def __init__(self, entry_df: pd.DataFrame() = None):
        """
        df - main part of Chromosome
        It should contain a row for each demand.
        And it should contain following columns:
            - CityA - source of demand
            - CityB - destination of demand
            - demand - value of demand in GpS
            - chosen_path - chosen path for demand
        """
        self.transponders_list = ['10G', '40G', '100G']
        self.transponders_cost = {
            '10G': 1,
            '40G': 3,
            '100G': 5
        }
        self.lambda_num = 96  # number of channels (wavelengths) on one connection between transponder pair
        self.df = entry_df.copy(deep=True)

        if 'chosen_path' not in self.df.columns:  # chromosome from initial population
            self.choose_random_path()

    def calculate_solution_cost(self):
        overall_cost = 0
        demand_per_edge = {}  # key - (CityA, CityB), value - sum of demands on this edge
        for index, row in self.df.iterrows():
            demand = row['Demand']
            chosen_path = row['chosen_path']
            for a, b in pairwise(chosen_path):  # iterate over pairs of cities
                if (a, b) not in demand_per_edge:
                    demand_per_edge[(a, b)] = 0
                demand_per_edge[(a, b)] += demand
        # print('demand_per_edge', demand_per_edge)
        # print('total demand on all edges', sum(demand_per_edge.values()))
        for (a, b), demand in demand_per_edge.items():
            transponders_set, cost = find_cheapest_set_of_transponders_for_edge(demand, self.transponders_cost)
            overall_cost += cost
        # print('overall_cost', overall_cost)
        return overall_cost

    def choose_random_path(self):
        # select randomly a path for each demand
        self.df['chosen_path'] = self.df['possible_paths'].apply(
            lambda possible_paths: random.SystemRandom().choice(possible_paths))
