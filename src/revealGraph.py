import networkx as nx
import matplotlib.pyplot as plt


def plotGraph(nodes_dict):
    G = nx.Graph()
    G.add_nodes_from(nodes_dict.keys())
    nx.draw_networkx(
        G, pos=nodes_dict, with_labels=True, node_color="#95C8D8", 
        font_size=8, font_color="#3E424B", font_weight="bold", node_size=50
    )
    plt.show()
