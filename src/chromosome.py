import pandas as pd
from random import choice, randint
from typing import List
from collections import Counter


def select_random_transformers_set(transponders_list: List[str]):
    random_choice = Counter([choice(transponders_list) for _ in range(randint(2, 4))])
    key_order = ['10G', '40G', '100G']
    return {k: random_choice[k] for k in key_order if k in random_choice}  # sort keys in transponders dict


class Chromosome:
    def __init__(self, edges: pd.DataFrame()):
        self.transponders_list = ['10G', '40G', '100G']
        self.transponders_cost = {
            '10G': 1,
            '40G': 3,
            '100G': 5
        }
        edges['transponders'] = [select_random_transformers_set(self.transponders_list) for _ in range(edges.shape[0])]
        self.df = edges

    def calculate_solution_cost(self):
        used_transponders = {'10G': 0, '40G': 0, '100G': 0}
        for index, t_set in self.df['transponders'].iteritems():
            for key, value in t_set.items():
                used_transponders[key] += value
        print('used_transponders', used_transponders)
        overall_cost = sum([self.transponders_cost[key] * value for key, value in used_transponders.items()])
        print('overall_cost', overall_cost)
        return overall_cost
