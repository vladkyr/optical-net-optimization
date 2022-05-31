import importGraph
import revealGraph

rel_path = './data/ger50_1m-1y/native.txt'
rows_to_skip = 9, 7, 7
network_nodes = 50
network_edges = 88
network_demands = 822 - 160


if __name__ == "__main__":
    nodes, edges, demands = importGraph.get_data_from_file(rel_path, network_nodes, network_edges, network_demands, rows_to_skip)
    print(demands)
    print("Summary demand to satisfy: ", sum(demands.loc[:, "Demand"]))
    revealGraph.plotGraph(nodes, edges)
