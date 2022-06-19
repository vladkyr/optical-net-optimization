import import_graph
import reveal_graph
from evolutionary_elgorithm import EvolutionaryAlgorithm
from optimizer import Optimizer

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
    print("Total demand to satisfy in entire net: ", sum(demands.loc[:, "Demand"]))

    # reveal_graph.plot_interactive_graph(edges)
    # reveal_graph.plot_graph(nodes, list(edges.itertuples(index=False, name=None)))

    optimization = True
    parallelize = True
    if not optimization:
        alg = EvolutionaryAlgorithm(edges=edges, demands=demands,
                                    cycles_no=5,
                                    mi_size=10, lambda_size=20,
                                    mutation_probability=0.5,
                                    gene_replacement_probability=0.5,
                                    number_of_paths_per_demand=2,
                                    select_method='RS')
        # Main algorithm's loop
        best_solution_found = alg.do_cycles()

        best_specimen, min_cost = alg.find_best_in_current_population()
        print('\nbest_specimen after all cycles completed:\n', best_specimen.df.columns)

        alg.change_of_min_cost_over_cycles()
        reveal_graph.plot_cost(alg.best_specimen_after_cycles)
    else:
        mi_set = [100]
        lambda_set = [10, 20, 50, 100, 200, 500]
        opt = Optimizer(mi_set=mi_set, lambda_set=lambda_set,
                        cycles_num=10, edges=edges, demands=demands)
        opt.optimize(parallelize=parallelize)
