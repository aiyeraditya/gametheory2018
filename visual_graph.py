import matplotlib.pyplot as plt
import networkx as nx
import random

# This function will plot a graph once it's given the node list, weights of edges as a 2D matrix.
# Also pass the title what you would like it saved as
def draw_graph(node_list, weights, name, saveas):
    G = nx.Graph()
    for node in node_list:
        G.add_node(node)
    pos=nx.circular_layout(G)
    nx.draw_networkx_nodes(G,pos,node_color='green',node_size=750)
    for i in range(len(node_list)):
        for j in range(i+1, len((node_list))):
            G.add_edge(node_list[i], node_list[j], weight = weights[i][j])      # Define the weights

    # Draw Graph with widths corresponding to weights
    for i in range(len(node_list)):
        for j in range(len(node_list)):
            nx.draw_networkx_edges(G, pos, edgelist = [(node_list[i],node_list[j])], width = weights[i][j]/15);
                # Draws the Graph edge wise with width corresponding to weights
    plt.axis('off')
    plt.title(name);
    plt.savefig(saveas + ".pdf")
    plt.show()


def birth_death(node_list, weights, cycle):
    # Birth death update rule
    return 1

# Function For Runing The Game
def evolve(node_list, weights, cycle):
    for i in range(cycle):
        birth_death(node_list, weights, cycle)
        print("Step", i + 1, "completed")
    # Rule for Evolution Here

node_list = [0, 1, 2]

#node_type tells whether the node is C (0) or D (1)
node_type = [0, 0, 1]

# Payoff matrix for two players
payoff_matrix = [[1, 0], [0, 1]]

weights = [[0, 1, 2], [1, 0, 3], [2,3,0]]

#instantiating the payoff matrix
payoff = [0, 0, 0]

# Computes payoffs fo each node
def compute_payoffs(node_list, weights):
    for node in node_list:
        payoff[node] = 0
        for i in node_list:
            payoff[node] += weights[node][i] * payoff_matrix[node_type[node]][node_type[i]]

compute_payoffs(node_list, weights)
print(payoff)

cycle = 3
evolve(node_list, weights, cycle)
#draw_graph(node_list, weights, "Title", "Saveas")
