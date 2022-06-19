from cmath import inf
import random
import networkx as nx

import pandas as pd
import numpy as np
from chromosome import Chromosome

'''
Original algorithm: Tomasz Pawlak
Adjustment to optical net problem: Vladyslav Kyryk
'''


def get_possible_paths_for_demand(graph: nx.Graph, city_a: str, city_b: str, k_paths: int):
    """
    Creates a set of k possible paths through which demand can be satisfied.
    Originally created to be used in pandas.DataFrame.apply function
    :param graph: networkx graph with defined edges between cities
    :param city_a: source
    :param city_b: destination
    :param k_paths: number of paths to create
    :return: list of possible paths for demand
    """
    shortest_paths = [p for p in nx.all_shortest_paths(G=graph, source=city_a, target=city_b)]
    if len(shortest_paths) >= k_paths:
        return shortest_paths[:k_paths]
    else:
        simple_paths = [p for p in nx.shortest_simple_paths(G=graph, source=city_a, target=city_b) if p not in shortest_paths]
        return shortest_paths + simple_paths[:k_paths-len(shortest_paths)]  # add to shortest_paths lacking amount of paths


class EvolutionaryAlgorithm:
    def __init__(
            self, edges: pd.DataFrame, demands: pd.DataFrame,
            cycles_no: int = 3,
            initial_population_size: int = 100, mutation_variance: float = 0.1,
            gene_replacement_probability: float = 0.5,
            select_method: str = 'TO',
            number_of_paths_per_demand: int = 2
    ):

        self.initial_population_size = initial_population_size
        self.cycles_no = cycles_no
        self.mutation_variance = mutation_variance
        self.gene_replacement_probability = gene_replacement_probability
        self.selection_method = select_method
        self.best_specimen_after_cycles = []

        # Create set of predefined paths for each demand
        self.demands = demands.copy(deep=True)
        net_graph = nx.from_pandas_edgelist(edges, source='CityA', target='CityB')
        self.demands['possible_paths'] = self.demands.apply(lambda demand: get_possible_paths_for_demand(net_graph,
                                                                                                         city_a=demand[
                                                                                                             'CityA'],
                                                                                                         city_b=demand[
                                                                                                             'CityB'],
                                                                                                         k_paths=number_of_paths_per_demand),
                                                            axis=1)
        # print('self.demands\n', self.demands)
        # print(self.demands.iloc[0]['possible_paths'])
        # print(self.demands.iloc[1]['possible_paths'])

        # Initiate population with random transponders set
        self.population = [Chromosome(self.demands) for _ in range(initial_population_size)]

    def selection(self):
        """
        Function cuts best points from selection and mutation in aim of 
        continue population of the best points
        """
        new_gen = []

        if self.selection_method == 'TO':  # Tournament Selection
            for i in range(int(len(self.population) / 4)):
                # Porównujemy 4 kolejne punkty, najlepszy z nich przechodzi
                # dalej i zostaje w populacji
                costs = []
                for j in range(4):
                    costs.append(self.population[4 * i + j].calculate_solution_cost())
                for j in range(4):
                    if costs[j] == min(costs):
                        new_gen.append(self.population[4 * i + j])
            return new_gen
        else:
            return (print(
                "Error: Selection methods\n",
                "TO - tournament selection\n"
            ))

    def cross(self):
        """
        Generuje nową populację krzyżując stare wartości osobników ze sobą, a
        potem przypisując ich wartości do nowych encji, które potem zostają
        dodane do tabeli nowej populacji punktów. Do encji dodajemy wartości
        połowy długości tabel aby uzyskać lepsze pomieszanie cech.
        """
        new_generation = []
        offspring_number = 2

        def apply_func(city_a, city_b, chosen_path_parent_x, parent_y, p_e):
            """
            p_e: probability of gene_replacement
            :return: list - chosen_path from one of the parents
            """
            if random.SystemRandom().uniform(0, 1) < p_e:
                return chosen_path_parent_x
            else:
                return parent_y.loc[
                    (parent_y['CityA'] == city_a) & (parent_y['CityB'] == city_b), 'chosen_path'].iloc[0]

        for i in range(self.initial_population_size*offspring_number):
            x = self.population[random.SystemRandom().randint(0, len(self.population) - 1)]
            y = self.population[random.SystemRandom().randint(0, len(self.population) - 1)]

            # copy DataFrame of parent x for offspring
            offspring_df = x.df.copy(deep=True)

            # insert transponders from y to x under certain probability
            offspring_df['transponders'] = offspring_df.apply(lambda e: apply_func(e['CityA'], e['CityB'],
                                                                                   e['chosen_path'], y.df,
                                                                                   self.gene_replacement_probability),
                                                              axis=1)

            # print('parent x:', x.df)
            # print('parent y:', y.df)
            # print('offspring:', offspring_df)
            new_generation.append(Chromosome(offspring_df))

        return new_generation

    def mutate(self):
        """
        Funkcja generuje odchyły od punktu, czyli zmiany w ilości 
        transponderów w chromosomie zgodnie z rozkładem normalnym 
        o wariancji równej mutation_variance
        """
        mutants = self.population.copy()
        for chromosome in mutants:
            chromosome.choose_random_path()
        return self.population + mutants

    def do_cycles(self):
        """
        Funkcja wykonuje pętle algorytmu ewolucyjnego i znajduje
        osobniki lepsze od poprzedniej generacji. Gens jest liczbą
        pokoleń następujących po inicjacji
        """
        for i in range(self.cycles_no):
            print('population size at cycle start', len(self.population))
            self.population = self.cross()
            self.population = self.mutate()
            self.population = self.selection()
            if i > 0 and i % int(self.cycles_no/10) == 0:
                print(f'completed cycle {i}')
                self.best_specimen_after_cycles.append((i, self.select_best_chromosome()[1]))

        return self.population

    def select_best_chromosome(self):
        min_cost = float('inf')  # set initial value of min cost to infinity
        best_chromosome = None
        for c in self.population:
            cost = c.calculate_solution_cost()
            if cost < min_cost:
                min_cost = cost
                best_chromosome = c
        return best_chromosome, min_cost

    def best_after_cycles(self):
        print(f"Change of min cost over cycles:\n", '\tcycle : cost')
        for i, cost in self.best_specimen_after_cycles:
            print("\t", i, ": ", cost)