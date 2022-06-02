import numpy as np 
import matplotlib.pyplot as plt
import os
'''
Tomasz Pawlak
304104

Temat 2.2

Zaimplementuj klasyczny algorytm ewolucyjny i porównaj wyniki dla różnych
metod selekcji (ruletkowa, progowa, turniejowa). Zaproponuj przykład
problemu optymalizacyjnego (inny niż na wykładzie) i zaprezentuj użycie
zaimplementowanego algorytmu do rozwiązania tego problemu.
'''


class Evolutionary_Alg:
    def __init__(
            self, select_method, range=100, cycles_no=3,
            population_size=100, mutation_c=10, target = [0,0]
    ):
        # Globalna definicja wymiarów wszystkich wykresów
        plt.rcParams['figure.figsize'] = [8, 6]

        # Definicja parametrów
        self.sm = select_method
        self.range = range
        self.population_size = population_size
        self.cycles_no = cycles_no
        self.mutation_c = mutation_c
        self.x1_range = [-self.range, self.range]
        self.x2_range = [-self.range, self.range]

        self.population = []

        # Definicja celu skupiania punktów jako dwuelementowa
        # tablica współrzędnych
        self.target = np.array(target)

        # Inicjacja populacji
        self.population = self.populate([self.x1_range, self.x2_range])
        self.draw()

        # Wykonanie pętli algorytmu
        self.do_cycles(self.cycles_no)

    def populate(self, features):
        '''
        Funckja inicjuje populację punktów ze składowymi
        z zakresu liczb rzeczywistych
        '''
        initial = []
        for i in range(self.population_size):
            entity = []
            for feature in features:
                val = np.random.randint(*feature)
                entity.append(val)
            initial.append(entity)

        return np.array(initial)

    def fitness(self):
        '''
        Funckja wybiera tą połowę punktów, która zgodnie z daną metodą
        selekcji nadają się do przedłużenia gatunku
        '''
        new_gen = []

        if self.sm == "RW":  # Selekcja Ruletkowa
            x = self.population.reshape(2*len(self.population))
            p = []
            # Zbiór omega jest zbiorem wszystkich możliwych promieni
            # poprowadzonych od wszystkich możliwych dostępnych targetów
            # Dla promieni większych niż omega uznajemy że prawdopodo-
            # bieństwo jest równe 0
            omega = self.range*np.sqrt(2)

            for i in range(len(self.population)):
                p_i = (omega-(np.sqrt((x[2*i]-self.target[0])**2 + (x[2*i+1]-self.target[1])**2))) / omega

                # Uniknięcie ujemnego prawdopodobieństwa
                if p_i < 0:
                    p_i = 0
                p.append(p_i)

            p = np.array(p)
            p /= np.sum(p)

            # Dodajemy do tablicy indeks punktu jeśli wylosujemy go z danym
            # prawdopodobieństwem z tablicy indeksów
            while len(new_gen) != len(self.population)/2:
                a = np.random.choice([i for i in range(len(p))], p=p)
                if (a not in new_gen):
                    new_gen.append(a)

            return np.array(new_gen)

        elif self.sm == "TH":  # Selekcja Progowa
            for index, entity in enumerate(self.population):
                new_gen1 = np.sum((entity-self.target)**2)
                new_gen.append((new_gen1, index))

            # Bierzemy pod uwagę tylko najlepsze przypadki
            new_gen = sorted(new_gen)[:int(self.population_size/2)]
            new_gen = np.array(new_gen)[:, 1]

            return new_gen

        elif self.sm == "TO":  # Selekcja Turniejowa
            x = self.population.reshape(2*len(self.population))
            for i in range(int(len(self.population)/2)):
                # Porównujemy dwa kolejne punkty, lepszy z nich przechodzi
                # dalej i zostaje w populacji
                dist1 = (x[4*i]-self.target[0])**2 + (x[4*i+1]-self.target[1])**2
                dist2 = (x[4*i+2]-self.target[0])**2 + (x[4*i+3]-self.target[1])**2

                new_gen.append(2*i) if (dist1 <= dist2) else new_gen.append(2*i+1)

            return np.array(new_gen)

        else:
            return(print(
                "Error: Selection methods\n",
                "RW - roulette wheel selection\n",
                "TH - threshold selection\n",
                "TO - tournament selection\n"
            ))

    def selection(self):
        '''
        Zwracamy punkty wybrane w funkcji fittest jako nadające się
        do reprodukcji
        '''
        fittest = self.fitness()

        new_generation = []

        for item in fittest:
            new_generation.append(self.population[item])

        return np.array(new_generation)

    def cross(self):
        '''
        Generuje nową populację krzyżując stare wartości osobników ze sobą, a
        potem przypisując ich wartości do nowych encji, które potem zostają
        dodane do tabeli nowej populacji punktów. Do encji dodajemy wartości
        połowy długości tabel aby uzyskać lepsze pomieszanie cech
        '''
        new_generation = []

        for i in range(self.population_size):
            x = self.population[np.random.randint(0, len(self.population))]
            y = self.population[np.random.randint(0, len(self.population))]

            entity = []
            entity.append(*x[:len(x)//2])
            entity.append(*y[len(y)//2:])

            new_generation.append(entity)

        return np.array(new_generation)

    def mutate(self):
        '''
        Funkcja generuje punkty z otoczenia danego punktu w promieniu
        podanym do inicjacji algorytmu jako mutacion_c - stała mutacji
        '''
        return self.population + np.random.randint(
            -self.mutation_c, self.mutation_c, 2*self.population_size
        ).reshape(self.population_size, 2)

    def do_cycles(self, gens=1):
        '''
        Funkcja wykonuje pętle algorytmu ewolucyjnego i znajduje
        osobniki lepsze od poprzedniej generacji. Gens jest liczbą
        pokoleń następujących po inicjacji
        '''
        for i in range(gens):
            self.population = self.selection()
            self.population = self.cross()
            self.population = self.mutate()

            self.draw()

        return self.population

    def draw(self):
        '''
        Funkcja generuje graf z modułu pyplot
        '''
        plt.xlim([-1*self.range, self.range])
        plt.ylim([-1*self.range, self.range])
        plt.scatter(
            self.population[:, 0], self.population[:, 1], c='green', s=12
        )
        plt.scatter(self.target[0], self.target[1], c='red', s=60)
        plt.show()


if __name__ == "__main__":
    '''
    # Typy selekcji:
    # RW - roulette wheel selection
    # TH - threshold selection
    # TO - tournament selection

    tmp1 = Evolutionary_Alg(
        range=100, select_method="TO", cycles_no=15,
        population_size=1000, mutation_c=10
        )

    tmp2 = Evolutionary_Alg(
        range=100, select_method="RW", cycles_no=5,
        population_size=1000, mutation_c=7
        )
    '''
    ealg = Evolutionary_Alg(
        range=100, select_method="TO", cycles_no=5,
        population_size=1000, mutation_c=10, target=[10, 10]
        )

    print('population', ealg.population)
