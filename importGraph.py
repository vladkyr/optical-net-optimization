import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

rel_path = 'ger50_1m-1y/demandMatrix-germany50-DFN-1month-200401.txt'
rows_to_skip = 19
network_nodes = 50

with open(rel_path, 'r') as f:
    # nodes = str(pd.read_csv(f, nrows=network_nodes, skiprows=19, header=None)).replace("(", "").replace(")", "")
    nodes = pd.read_csv(f, nrows=network_nodes, skiprows=rows_to_skip, header=None, delimiter=" ")
    nodes.drop([0, 1, 3, 6], inplace=True, axis=1)
    nodes.columns = ["City", "X", "Y"]
    nodes["City"].astype("string")
    nodes["X"].astype("float")
    nodes["Y"].astype("float")

nodes_dict = {}
for i in range(network_nodes):
    nodes_dict[nodes.loc[:, "City"].values[i]] = [tuple(x) for x in (nodes.loc[:, ["X", "Y"]]).values][i]

G = nx.Graph()
G.add_nodes_from(nodes_dict.keys())
nx.draw_networkx(G, pos=nodes_dict, with_labels=True, node_color="#95C8D8", font_size=8, font_color="#3E424B", node_size=50)

ax = plt.axes([0, 0, 15, 15])
ax.set_aspect('equal')

plt.show()
