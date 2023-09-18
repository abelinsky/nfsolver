import matplotlib.pyplot as plt
import networkx as nx


def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G, seed=42),
        with_labels=True,
        node_color=color,
        cmap="Set2",
    )
    plt.show()
