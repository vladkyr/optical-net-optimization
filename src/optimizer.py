import time

import pandas as pd
from typing import List
import multiprocessing as mp

from evolutionary_elgorithm import EvolutionaryAlgorithm
import reveal_graph


class Optimizer:
    def __init__(self,
                 mi_set: List[int], lambda_set: List[int],
                 cycles_num: int,
                 edges: pd.DataFrame, demands: pd.DataFrame,
                 plots_folder_name: str):
        self.mi_set = mi_set
        self.lambda_set = lambda_set
        self.cycles_num = cycles_num
        self.edges = edges
        self.demands = demands
        self.plots_folder_name = plots_folder_name

    def run_algorithm(self, mi: int, lmbd: int):
        alg = EvolutionaryAlgorithm(edges=self.edges, demands=self.demands,
                                    cycles_no=self.cycles_num,
                                    mi_size=mi, lambda_size=lmbd,
                                    mutation_probability=0.75,
                                    gene_replacement_probability=0.5,
                                    number_of_paths_per_demand=2,
                                    select_method='RS')
        # Main algorithm's loop
        alg.do_cycles()

        # alg.change_of_min_cost_over_cycles()
        reveal_graph.plot_cost(alg.best_specimen_after_cycles, file_name=f'{self.plots_folder_name}/Plot_mi{mi}_lmbd{lmbd}')
        return alg.best_specimen_after_cycles

    def optimize(self, parallelize: bool = True):
        if parallelize:
            with mp.Pool(mp.cpu_count()) as pool:
                results = [(pool.apply_async(self.run_algorithm, args=(mi, lmbd)), mi, lmbd)
                           for mi in self.mi_set
                           for lmbd in self.lambda_set]
                while False in [r.ready() for r, _, _ in results]:
                    print('waiting 1 more minute for tasks to finish...')
                    time.sleep(60)
                results = [(r.get(), m, l) for r, m, l in results]
                print('results', results)
            reveal_graph.plot_multiple_lines(results, file_name=f'{self.plots_folder_name}/Plot_complete')
        else:
            plot_data = []
            for mi in self.mi_set:
                for lmbd in self.lambda_set:
                    print(f'------------- mi {mi}, lambda {lmbd} -----------------')
                    cost_over_cycles = self.run_algorithm(mi=mi, lmbd=lmbd)
                    plot_data.append((cost_over_cycles, mi, lmbd))
            reveal_graph.plot_multiple_lines(plot_data, file_name=f'{self.plots_folder_name}/Plot_complete')
