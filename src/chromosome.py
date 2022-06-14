import pandas as pd
from random import choice, randint
from typing import List
from collections import Counter


def select_random_transformers_set(transponders_list: List[str]):
    random_choice = Counter([choice(transponders_list) for _ in range(randint(2, 4))])
    key_order = ['10G', '40G', '100G']
    return {k: random_choice[k] for k in key_order if k in random_choice}  # sort keys in transponders dict


def create_random_chromosome(links: List[str], transponders: List[str]) -> pd.DataFrame:
    return pd.DataFrame({'links': links, 'transponders': [select_random_transformers_set(transponders) for _ in links]})


def create_initial_population(links: List[str], transponders: List[str], population_size: int) -> List[pd.DataFrame]:
    return [create_random_chromosome(links, transponders) for _ in range(population_size)]
