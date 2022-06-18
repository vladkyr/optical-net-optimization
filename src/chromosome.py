import pandas as pd
import random
from typing import List
from collections import Counter


def select_random_transformers_set(transponders_list: List[str]):
    random_choice = Counter(
        [random.SystemRandom().choice(transponders_list) for _ in range(random.SystemRandom().randint(2, 4))])
    key_order = ['10G', '40G', '100G']
    return {k: random_choice[k] for k in key_order if k in random_choice}  # sort keys in transponders dict


class Chromosome:
    def __init__(self, entry_df: pd.DataFrame() = None):
        self.transponders_list = ['10G', '40G', '100G']
        self.transponders_cost = {
            '10G': 1,
            '40G': 3,
            '100G': 5
        }
        self.lambda_num = 96  # number of channels (wavelengths) on one connection between transponder pair
        self.df = entry_df.copy(deep=True)
        if 'transponders' not in self.df.columns:
            self.df['transponders'] = [select_random_transformers_set(self.transponders_list) for _ in
                                       range(entry_df.shape[0])]

    def calculate_solution_cost(self):
        used_transponders = {'10G': 0, '40G': 0, '100G': 0}
        for index, t_set in self.df['transponders'].iteritems():
            for key, value in t_set.items():
                used_transponders[key] += value
        # print('used_transponders', used_transponders)
        overall_cost = sum([self.transponders_cost[key] * value for key, value in used_transponders.items()])
        # print('overall_cost', overall_cost)
        return overall_cost
