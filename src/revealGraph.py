import networkx as nx
import matplotlib.pyplot as plt


def plotGraph(nodes_dict, edges_list):
    G = nx.Graph()
    G.add_nodes_from(nodes_dict.keys())
    G.add_edges_from(edges_list)
    nx.draw_networkx(
        G, pos=nodes_dict, with_labels=True, node_color="#95C8D8", 
        font_size=8, font_color="#3E424B", font_weight="bold", node_size=50
    )
    plt.show()
