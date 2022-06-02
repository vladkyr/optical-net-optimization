import pandas as pd
from random import choice, randint
from typing import List


def create_initial_population(links: List[str], transponders: List[str], population_size: int) -> List[pd.DataFrame]:
    def create_random_chromosome(links: List[str]) -> pd.DataFrame:
        return pd.DataFrame({'links': links, 'transponders': [[choice(transponders) for _ in range(randint(2, 4))] for _ in links]})
        
    return [create_random_chromosome(links) for _ in range(population_size)]