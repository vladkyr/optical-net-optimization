import random
import networkx as nx
import pandas as pd
from typing import List

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
        simple_paths = [p for p in nx.shortest_simple_paths(G=graph, source=city_a, target=city_b) if
                        p not in shortest_paths]
        return shortest_paths + simple_paths[
                                :k_paths - len(shortest_paths)]  # add to shortest_paths lacking amount of paths


class EvolutionaryAlgorithm:
    def __init__(
            self, edges: pd.DataFrame, demands: pd.DataFrame,
            cycles_no: int = 3, mi_size: int = 10, lambda_size: int = 10,
            mutation_probability: float = 0.5,
            gene_replacement_probability: float = 0.5,
            number_of_paths_per_demand: int = 2,
            select_method: str = 'RS'
    ):
        self.cycles_no = cycles_no
        self.mi_size = mi_size  # µ - algorithm hyperparameter
        self.lambda_size = lambda_size  # λ - algorithm hyperparameter
        self.mutation_probability = mutation_probability
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

        # Initiate population with random transponders set
        self.population = [Chromosome(self.demands) for _ in range(self.mi_size)]
        self.best_so_far = self.find_best_in_current_population()  # best_so_far - tuple of (Chromosome, cost)
        print('Cost of best solution in initial population:', self.best_so_far[1])

    def selection(self):
        """
        Function cuts best points from selection and mutation in aim of 
        continue population of the best points
        """

        if self.selection_method == 'RS':  # Random Selection
            return random.SystemRandom().choices(self.population, k=self.lambda_size)
        elif self.selection_method == 'TO':  # Tournament Selection
            new_gen = []
            print('population size', len(self.population))
            for i in range(int(len(self.population) / 4)):
                # Porównujemy 4 kolejne punkty, najlepszy z nich przechodzi
                # dalej i zostaje w populacji
                costs = []
                for j in range(4):
                    costs.append(self.population[4 * i + j].calculate_solution_cost()[0])
                for j in range(4):
                    if costs[j] == min(costs):
                        new_gen.append(self.population[4 * i + j])
                        break
            print('new_gen size', len(new_gen))
            return new_gen
        else:
            return (print(
                "Error: Selection methods\n",
                "TO - tournament selection\n"
            ))

    def cross(self, offspring: List[Chromosome]):
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

        for i in range(self.lambda_size):
            x = offspring[random.SystemRandom().randint(0, len(offspring) - 1)]
            y = offspring[random.SystemRandom().randint(0, len(offspring) - 1)]

            # copy DataFrame of parent x for offspring
            child_df = x.df.copy(deep=True)

            # insert transponders from y to x under certain probability
            child_df['chosen_path'] = child_df.apply(lambda e: apply_func(e['CityA'], e['CityB'],
                                                                          e['chosen_path'], y.df,
                                                                          self.gene_replacement_probability),
                                                     axis=1)

            # print('parent x:', x.df)
            # print('parent y:', y.df)
            # print('child:', child_df)
            new_generation.append(Chromosome(child_df))

        return new_generation

    def mutate(self, new_generation: List[Chromosome]):
        """
        Funkcja generuje odchyły od punktu, czyli zmiany w ilości 
        transponderów w chromosomie zgodnie z rozkładem normalnym 
        o wariancji równej mutation_variance
        """
        mutants = new_generation.copy()
        for chromosome in mutants:
            # each demand is mutated separately with predefined mutation_probability
            if random.SystemRandom().uniform(0, 1) < self.mutation_probability:
                chromosome.choose_random_path()
        return mutants

    def select_best_chromosomes_for_new_population(self, mutants: List[Chromosome]):
        pairs = []
        for c in self.population + mutants:
            pairs.append((c, c.calculate_solution_cost()[0]))
        pairs.sort(key=lambda x: x[1], reverse=True)
        self.population = [p[0] for p in pairs][:self.mi_size]
        current_best, current_best_cost = pairs[0]
        # update best solution
        if current_best_cost < self.best_so_far[1]:
            self.best_so_far = (current_best, current_best_cost)

    def do_cycles(self):
        """
        Funkcja wykonuje pętle algorytmu ewolucyjnego i znajduje
        osobniki lepsze od poprzedniej generacji. Gens jest liczbą
        pokoleń następujących po inicjacji
        """
        for i in range(self.cycles_no):
            # print('population size at cycle start', len(self.population))
            offspring = self.selection()
            crossed_offspring = self.cross(offspring)
            mutants = self.mutate(crossed_offspring)
            self.select_best_chromosomes_for_new_population(mutants)

            if self.cycles_no < 10 or (i % int(self.cycles_no / 10) == 0):
                print(f'completed cycle {i}')
                self.best_specimen_after_cycles.append((i, self.best_so_far[1]))

        return self.best_so_far

    def find_best_in_current_population(self):
        min_cost = float('inf')  # set initial value of min cost to infinity
        best_chromosome = None
        for c in self.population:
            cost, _ = c.calculate_solution_cost()
            if cost < min_cost:
                min_cost = cost
                best_chromosome = c
        return best_chromosome, min_cost

    def change_of_min_cost_over_cycles(self):
        print(f"\nChange of min cost over cycles:\n", '\tcycle : cost')
        for i, cost in self.best_specimen_after_cycles:
            print("\t", i, ": ", cost)
