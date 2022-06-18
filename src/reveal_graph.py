import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt

graph_file = 'graph.html'


def plot_interactive_graph(edges):
    nx_graph = nx.from_pandas_edgelist(edges, source='CityA', target='CityB')
    net = Network('500px', '500px')
    net.from_nx(nx_graph)
    # net.show(graph_file)
    net.save_graph(graph_file)


def plot_graph(nodes, edges):
    G = nx.Graph()

    nodes_dict = {}
    for i in range(nodes.shape[0]):
        nodes_dict[nodes.loc[:, "City"].values[i]] = [tuple(x) for x in (nodes.loc[:, ["X", "Y"]]).values][i]
    G.add_nodes_from(nodes_dict.keys())

    G.add_edges_from(edges)
    nx.draw_networkx(
        G, pos=nodes_dict, with_labels=True, node_color="#95C8D8",
        font_size=8, font_color="#3E424B", font_weight="bold", node_size=50
    )
    plt.show()
