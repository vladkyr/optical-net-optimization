import import_graph
import reveal_graph
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

    nodes, edges, demands = import_graph.get_data_from_file(rel_path, network_nodes, network_edges,
                                                            network_demands, rows_to_skip)
    # print('nodes', nodes)
    # print('edges', edges)
    # print('demands', demands)
    print("Total demand to satisfy in entire net: ", sum(demands.loc[:, "Demand"]))

    reveal_graph.plot_interactive_graph(edges)
    # reveal_graph.plot_graph(nodes, list(edges.itertuples(index=False, name=None)))

    alg = EvolutionaryAlgorithm(edges=edges, demands=demands,
                                cycles_no=500,
                                initial_population_size=10, mutation_variance=0.2,
                                gene_replacement_probability=0.5,
                                select_method='TO',
                                number_of_paths_per_demand=3)
    # Main algorithm's loop
    final_population = alg.do_cycles()

    best_specimen, min_cost = alg.select_best_chromosome()
    print('best_specimen after all cycles completed:\n', best_specimen.df)

    alg.best_after_cycles()
