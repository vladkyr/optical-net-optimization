import networkx as nx
import pandas as pd
from pyvis.network import Network
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import time

graph_file = 'graph.html'


def plot_interactive_graph(edges: pd.DataFrame):
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


def plot_cost(cost_list: List[Tuple[int, int]], file_name: str = ''):
    cost_list = np.array(cost_list)
    plt.plot(cost_list[:, 0], cost_list[:, 1], color='g',
             linestyle="dashed", linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Cost of solution")
    plt.title("Change of min cost over cycles")
    if file_name == '':
        file_name = f'Costs_plot_{time.time()-1656000000}'
    plt.savefig(f'./plots/{file_name}.png')
    # plt.show()


def plot_multiple_lines(cost_lists: List[Tuple[List[Tuple[int, int]], int, int]], file_name: str = ''):
    for cl, mi, lmbd in cost_lists:
        cl = np.array(cl)
        plt.plot(cl[:, 0], cl[:, 1], linewidth=2, label=f'mi={mi}, lmbd={lmbd}')
    plt.xlabel("Iterations")
    plt.ylabel("Cost of solution")
    plt.title("Change of min cost over cycles")
    if file_name == '':
        file_name = f'Costs_plot_{time.time()-1656000000}'
    plt.savefig(f'./plots/{file_name}.png')
    plt.show()
