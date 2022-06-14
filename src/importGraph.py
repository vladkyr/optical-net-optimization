import pandas as pd


def get_data_from_file(rel_path, network_nodes, network_edges, network_demands, rows_to_skip):
    with open(rel_path, 'r') as f:
        nodes = pd.read_csv(f, nrows=network_nodes, skiprows=rows_to_skip[0] - 1, header=None, delimiter=" ")
    nodes = nodes[2].to_list()

    with open(rel_path, 'r') as f:
        edges = pd.read_csv(
            f, nrows=network_edges, skiprows=rows_to_skip[0] + rows_to_skip[1] + network_nodes - 1,
            header=None, delimiter=" ")
    edges = edges[[4, 5]]
    edges.columns = ["CityA", "CityB"]
    edges["CityA"].astype("string")
    edges["CityB"].astype("string")

    with open(rel_path, 'r') as f:
        demands = pd.read_csv(
            f, nrows=network_demands,
            skiprows=rows_to_skip[0] + rows_to_skip[1] + rows_to_skip[2] + network_nodes + network_edges - 1,
            header=None, delimiter=" ")
    demands = demands[[4, 5, 8]]
    demands.columns = ["CityA", "CityB", "Demand"]
    demands["CityA"].astype("string")
    demands["CityB"].astype("string")
    demands["Demand"].astype("float")

    return nodes, edges, demands
