import numpy as np
import matplotlib.pyplot as plt
import random

from chromosome import Chromosome

'''
Original algorithm: Tomasz Pawlak
Adjustment to optical net problem: Vladyslav Kyryk
'''


class EvolutionaryAlgorithm:
    def __init__(
            self, select_method, edges, range_r=100, cycles_no=3,
            population_size=100, mutation_c=10, target=[0, 0],
            gene_replacement_probability=0.5
    ):
        # Globalna definicja wymiarów wszystkich wykresów
        plt.rcParams['figure.figsize'] = [8, 6]

        # Definicja parametrów
        self.sm = select_method
        self.range = range_r
        self.population_size = population_size
        self.cycles_no = cycles_no
        self.mutation_c = mutation_c
        self.gene_replacement_probability = gene_replacement_probability

        # Definicja celu skupiania punktów jako dwuelementowa
        # tablica współrzędnych
        self.target = np.array(target)

        # Inicjacja populacji z losowym zestawem transponderów
        self.population = []
        for _ in range(population_size):
            new_c = Chromosome(edges)
            # print('edges', edges)
            self.population.append(new_c)
        #         self.draw()

        # Wykonanie pętli algorytmu
        self.do_cycles(self.cycles_no)

    def selection(self):
        """
        Funckja wybiera tą połowę punktów, która zgodnie z daną metodą
        selekcji nadają się do przedłużenia gatunku
        """
        new_gen = []

        if self.sm == "TO":  # Selekcja Turniejowa
            for i in range(int(len(self.population) / 2)):
                # Porównujemy dwa kolejne punkty, lepszy z nich przechodzi
                # dalej i zostaje w populacji
                cost1 = self.population[2 * i].calculate_solution_cost()
                cost2 = self.population[2 * i + 1].calculate_solution_cost()

                new_gen.append(self.population[2 * i]) if (cost1 <= cost2) else new_gen.append(
                    self.population[2 * i + 1])
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

            :param transponders_set:
            :param city_b:
            :param city_a:
            :param row:
            :param other_parent:
            :param p_e: probability of gene_replacement
            :return:
            """
            if random.SystemRandom().uniform(0, 1) < p_e:
                return transponders_set
            else:
                return other_parent.loc[(other_parent['CityA'] == city_a) & (other_parent['CityB'] == city_b), 'transponders'].iloc[0]

        for i in range(self.population_size):
            print('population len', len(self.population))
            x = self.population[random.SystemRandom().randint(0, len(self.population)-1)]
            y = self.population[random.SystemRandom().randint(0, len(self.population)-1)]

            # print('parent x:', x.df)
            # print('parent y:', y.df)

            # copy DataFrame of parent x for offspring
            offspring_df = x.df

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
        Funkcja generuje punkty z otoczenia danego punktu w promieniu
        podanym do inicjacji algorytmu jako mutacion_c - stała mutacji
        """
        print('Mutacja jeszcze nie zaimplementowana!')
        return self.population
        # return self.population + np.random.randint(
        #     -self.mutation_c, self.mutation_c, 2*self.population_size
        # ).reshape(self.population_size, 2)

    def do_cycles(self, gens=1):
        """
        Funkcja wykonuje pętle algorytmu ewolucyjnego i znajduje
        osobniki lepsze od poprzedniej generacji. Gens jest liczbą
        pokoleń następujących po inicjacji
        """
        for i in range(gens):
            self.population = self.selection()
            self.population = self.cross()
            self.population = self.mutate()

            # self.draw()

        return self.population

    def draw(self):
        """
        Funkcja generuje graf z modułu pyplot
        """
        plt.xlim([-1 * self.range, self.range])
        plt.ylim([-1 * self.range, self.range])
        plt.scatter(
            self.population[:, 0], self.population[:, 1], c='green', s=12
        )
        plt.scatter(self.target[0], self.target[1], c='red', s=60)
        plt.show()

    def select_best_chromosome(self):
        min_cost = 99999999
        best_chromosome = None
        for c in self.population:
            cost = c.calculate_solution_cost()
            if cost < min_cost:
                min_cost = cost
                best_chromosome = c
        return best_chromosome, min_cost


if __name__ == "__main__":
    '''
    # Typy selekcji:
    # TO - tournament selection

    ealg = Evolutionary_Algorithm(
        #     range=100, select_method="TO", cycles_no=5,
        #     population_size=1000, mutation_c=10, target=[10, 10]
        #     )
    '''

    # ealg = Evolutionary_Algorithm(
    #     range=100, select_method="TO", cycles_no=5,
    #     population_size=1000, mutation_c=10, target=[10, 10]
    #     )
    #
    # print('population', ealg.population)
