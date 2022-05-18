import pandas as pd


def get_nodes(rel_path, network_nodes, rows_to_skip):
    with open(rel_path, 'r') as f:
        nodes = pd.read_csv(f, nrows=network_nodes, skiprows=rows_to_skip, header=None, delimiter=" ")
        nodes.drop([0, 1, 3, 6], inplace=True, axis=1)
        nodes.columns = ["City", "X", "Y"]
        nodes["City"].astype("string")
        nodes["X"].astype("float")
        nodes["Y"].astype("float")

    nodes_dict = {}
    for i in range(network_nodes):
        nodes_dict[nodes.loc[:, "City"].values[i]] = [tuple(x) for x in (nodes.loc[:, ["X", "Y"]]).values][i]

    return nodes_dict
