import importGraph
import revealGraph

rel_path = '../data/ger50_1m-1y/demandMatrix-germany50-DFN-1month-200401.txt'
rows_to_skip = 19, 83
network_nodes = 50      # 69-20+1
network_demands = 2186 - 84 + 1  # 2186-84+1

if __name__ == "__main__":
    nodes = importGraph.get_nodes(rel_path, network_nodes, rows_to_skip[0])
    revealGraph.plotGraph(nodes)
