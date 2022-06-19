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
    # TODO: implement apply function
    return []


class EvolutionaryAlgorithm:
    def __init__(
            self, edges: pd.DataFrame, demands: pd.DataFrame,
            range_r: int = 100, cycles_no: int = 3,
            population_size: int = 100, mutation_variance: float = 0.1,
            gene_replacement_probability: float = 0.5,
            select_method: str = 'TO',
            number_of_paths_per_demand: int = 3
    ):

        self.range = range_r
        self.population_size = population_size
        self.cycles_no = cycles_no
        self.mutation_variance = mutation_variance
        self.gene_replacement_probability = gene_replacement_probability
        self.selection_method = select_method
        self.best_specimen_after_cicles = []

        # Create set of predefined paths for each demand
        self.demands = demands.copy(deep=True)
        net_graph = nx.from_pandas_edgelist(edges, source='CityA', target='CityB')
        self.demands['possible_paths'] = self.demands.apply(lambda demand: get_possible_paths_for_demand(net_graph,
                                                                                                         city_a=demand['CityA'],
                                                                                                         city_b=demand['CityB'],
                                                                                                         k_paths=number_of_paths_per_demand),
                                                            axis=1)
        print('self.demands\n', self.demands)

        # Initiate population with random transponders set
        self.population = [Chromosome(edges) for _ in range(population_size)]

        # Main algorithm's loop
        self.do_cycles(self.cycles_no)

    def selection(self):
        """
        Function cuts best points from selection and mutation in aim of 
        continue population of the best points
        """
        new_gen = []

        if self.selection_method == 'TO':  # Selekcja Turniejowa
            for i in range(int(len(self.population) / 4)):
                # Porównujemy 4 kolejne punkty, najlepszy z nich przechodzi
                # dalej i zostaje w populacji
                costs = []
                for j in range(4):
                    costs.append(self.population[4 * i + j].calculate_solution_cost())
                for j in range(4):
                    if costs[j] == min(costs):
                        new_gen.append(self.population[4 * i + j])
                        break
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

        def apply_func(city_a, city_b, transponders_set, other_parent, p_e):
            """
            p_e: probability of gene_replacement
            :return: dict - set of transponders with quantities
            """
            if random.SystemRandom().uniform(0, 1) < p_e:
                return transponders_set
            else:
                return other_parent.loc[(other_parent['CityA'] == city_a) & (other_parent['CityB'] == city_b), 'transponders'].iloc[0]

        for i in range(self.population_size):
            x = self.population[random.SystemRandom().randint(0, len(self.population)-1)]
            y = self.population[random.SystemRandom().randint(0, len(self.population)-1)]

            # print('parent x:', x.df)
            # print('parent y:', y.df)

            # copy DataFrame of parent x for offspring
            offspring_df = x.df.copy(deep=True)

            # insert transponders from y to x under certain probability
            offspring_df['transponders'] = offspring_df.apply(lambda e: apply_func(e['CityA'], e['CityB'],
                                                                                   e['transponders'], y.df,
                                                                                   self.gene_replacement_probability),
                                                              axis=1)

            # print('offspring:', offspring_df)
            new_generation.append(Chromosome(offspring_df))

        return new_generation

    def mutate(self):
        """
        Funkcja generuje odchyły od punktu, czyli zmiany w ilości 
        transponderów w chromosomie zgodnie z rozkładem normalnym 
        o wariancji równej mutation_variance
        """
        mutants = self.population
        for i in range(len(self.population)):
            for j in mutants[i].df["transponders"]:
            
                j["10G"] += int(np.random.normal(0,self.mutation_variance))
                j["40G"] += int(np.random.normal(0,self.mutation_variance))
                j["100G"]+= int(np.random.normal(0,self.mutation_variance))
            mutants.append(self.population[i])

        return self.population + mutants

    def do_cycles(self, gens=1):
        """
        Funkcja wykonuje pętle algorytmu ewolucyjnego i znajduje
        osobniki lepsze od poprzedniej generacji. Gens jest liczbą
        pokoleń następujących po inicjacji
        """
        for i in range(gens):
            self.population = self.cross()
            self.population = self.mutate()
            self.population = self.selection()
            if i%(self.cycles_no/10) == (self.cycles_no/10) - 1:
                self.best_specimen_after_cicles.append(self.select_best_chromosome())

        return self.population

    def select_best_chromosome(self):
        min_cost = 99999999
        best_chromosome = None
        for c in self.population:
            cost = c.calculate_solution_cost()
            if cost < min_cost:
                min_cost = cost
                best_chromosome = c
        return best_chromosome, min_cost

    def bests_after_cycles(self):
        print("Min Cost after cycles:", len(self.best_specimen_after_cicles))
        for i in range(10):
            print("\t", int((i+1)*self.cycles_no/10), ": ", self.best_specimen_after_cicles[i][1])