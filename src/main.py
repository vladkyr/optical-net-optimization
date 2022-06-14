import importGraph
import revealGraph
from evolutionary_elgorithm import EvolutionaryAlgorithm

# rel_path = './data/germany50/native.txt'
rel_path = './data/germany17/native.txt'
if 'germany50' in rel_path:
    rows_to_skip = 9, 7, 7
    network_nodes = 50
    network_edges = 88
    network_demands = 822 - 160
elif 'germany17' in rel_path:
    rows_to_skip = 9, 7, 7
    network_nodes = 17
    network_edges = 26
    network_demands = 121
else:
    raise ValueError

if __name__ == "__main__":

    nodes, edges, demands = importGraph.get_data_from_file(rel_path, network_nodes, network_edges,
                                                           network_demands, rows_to_skip)
    # print('nodes', nodes)
    # print('edges', edges)
    # print('demands', demands)
    print("Total demand to satisfy in entire net: ", sum(demands.loc[:, "Demand"]))
    # revealGraph.plotGraph(nodes, edges)

    links = [f'{e[0]}_{e[1]}' for e in edges]

    alg = EvolutionaryAlgorithm(links=links, select_method="TO", range=100, cycles_no=5,
                                population_size=3, mutation_c=10, target=[10, 10])

    print('EvolutionaryAlgorithm', alg)

    print('population', alg.population)

    alg.calculate_solution_cost(alg.population[0])
