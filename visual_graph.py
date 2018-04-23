import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

# This function will plot a graph once it's given the node list, weights of edges as a 2D matrix.
# Also pass the title what you would like it saved as
def draw_graph(node_list, weights, name, saveas):
    G = nx.Graph()
    for node in node_list:
        G.add_node(node)
    pos=nx.circular_layout(G)

    node_colors = ['#ff00ff' if node == 1 else '#00b8f8' for node in node_type]
    nx.draw_networkx_nodes(G,pos,node_color=node_colors, node_size=750)
    for i in range(len(node_list)):
        for j in range(i+1, len((node_list))):
            G.add_edge(node_list[i], node_list[j], weight = weights[i][j])  # Define the weights

    # Draw Graph with widths corresponding to weights
    for i in range(len(node_list)):
        for j in range(len(node_list)):
            nx.draw_networkx_edges(G, pos, edgelist = [(node_list[i],node_list[j])], width = weights[i][j])
                # Draws the Graph edge wise with width corresponding to weights

    arc_weight = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=arc_weight, font_color='black', label_pos=0.3)
    plt.axis('off')
    #plt.title(name);
    plt.savefig(saveas + ".pdf")
    #plt.show()

# Birth death update rule
def birth_death(node_list, weights, cycle):
    # Select a node to reproduce at random
    node_to_reproduce = random.choice(node_list)
    print("Node", node_to_reproduce, "will reproduce")
    prob_dist = list(transformed_payoff)
    for node in node_list:
        prob_dist[node] = weights[node_to_reproduce][node] / transformed_payoff[node]
    normalization_factor = np.sum(prob_dist)
    # create a probability distribution proportional to e_ij / F_j
    for node in node_list:
        normalized_prob_dist = [i/normalization_factor for i in prob_dist]
    print("The probability distribution to choose neighbour to be killed is", prob_dist)
    print("The normalized probability distribution to choose neighbour to be killed is", normalized_prob_dist)
    # choose a node to die based on the probability distribution
    node_to_die_array = np.random.choice(node_list, 1, False, normalized_prob_dist)
    node_to_die = node_to_die_array[0]
    print("Node", node_to_die, "will die")

    node_type[node_to_die] = node_type[node_to_reproduce]

    for i in range(len(weights)):
        weights[node_to_die][i] = weights[node_to_reproduce][i]

    for i in range(len(weights)):
        weights[i][node_to_die] = weights[i][node_to_reproduce]
        weights[node_to_die][i] = weights[node_to_reproduce][i]

    weights[node_to_die][node_to_reproduce] = 1
    weights[node_to_reproduce][node_to_die] = 1
    update_payoffs(node_list, weights)

def death_birth(node_list, weights, cycle):

    death_prob_dist = list(transformed_payoff)
    for node in node_list:
        death_prob_dist[node] = 1 / transformed_payoff[node]
    death_normalization_factor = np.sum(death_prob_dist)

    for node in node_list:
        normalized_death_dist = [i/death_normalization_factor for i in death_prob_dist]
    node_to_die_array = np.random.choice(node_list, 1, False, normalized_death_dist)
    node_to_die = node_to_die_array[0]

    print("Node", node_to_die, "will die")

    reproduce_prob_dist = list(transformed_payoff)
    for node in node_list:
        reproduce_prob_dist[node] = weights[node_to_die][node]
    reproduce_normalization_factor = np.sum(reproduce_prob_dist)
    # create a probability distribution proportional to e_ij
    for node in node_list:
        normalized_reproduce_dist = [i/reproduce_normalization_factor for i in reproduce_prob_dist]
    print("The probability distribution to choose neighbour to be reproduce is", reproduce_prob_dist)
    print("The normalized probability distribution to choose neighbour to reproduce is", normalized_reproduce_dist)
    # choose a node to die based on the probability distribution
    node_to_reproduce_array = np.random.choice(node_list, 1, False, normalized_reproduce_dist)
    node_to_reproduce = node_to_reproduce_array[0]
    print("Node", node_to_die, "will reproduce")

    node_type[node_to_die] = node_type[node_to_reproduce]

    for i in range(len(weights)):
        weights[node_to_die][i] = weights[node_to_reproduce][i]

    for i in range(len(weights)):
        weights[i][node_to_die] = weights[i][node_to_reproduce]

    weights[node_to_die][node_to_reproduce] = 1
    weights[node_to_reproduce][node_to_die] = 1
    update_payoffs(node_list, weights)

# Function For Runing The Game
def evolve(node_list, weights, cycle):
    for i in range(cycle):

        print("The node types are", node_type)
        print("The total payoffs are", payoff)
        print("The transformed payoffs are", transformed_payoff, "\n")
        print("The weights are", weights, "\n")
        plt.close()
        draw_graph(node_list, weights, "Stage %d" % i, "stage %d" % i)
        birth_death(node_list, weights, cycle)
        print("Step", i + 1, "completed\n")


    # Rule for Evolution Here
node_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# node_type tells whether the node is C (0) or D (1)
node_type = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

# Payoff matrix for two players
payoff_matrix = [[3, 1], [4, 2]]

weights = [ [0, 1, 2, 1, 4, 2, 5, 6, 4, 3],
            [1, 0, 3, 4, 3, 0, 1, 3, 1, 5],
            [2, 3, 0, 0, 2, 5, 4, 2, 3, 0],
            [1, 4, 0, 0, 1, 0, 0, 0, 0, 0],
            [4, 3, 2, 1, 0, 3, 4, 2, 1, 0],
            [2, 0, 5, 0, 3, 1, 2, 1, 2, 1],
            [5, 1, 4, 0, 4, 2, 4, 3, 1, 2],
            [6, 3, 2, 0, 2, 1, 3, 0, 0, 0],
            [4, 1, 3, 0, 1, 2, 1, 0, 3, 2],
            [3, 5, 0, 0, 0, 1, 2, 0, 2, 5] ]
#weights = [[1 for i in range(5)] for i in range(5)]
#instantiating the payoff matrix f_j
payoff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# F_j
transformed_payoff = list(payoff)

# transformation function to scale payoffs - could be exponential
def transform(i):
    return i + 1

# Computes payoffs of each node
def update_payoffs(node_list, weights):
    for node in node_list:
        payoff[node] = 0
        for i in node_list:
            payoff[node] += weights[node][i] * payoff_matrix[node_type[node]][node_type[i]]

    for i in range(len(payoff)):
        transformed_payoff[i] =  transform(payoff[i])

update_payoffs(node_list, weights)
cycle = 20
evolve(node_list, weights, cycle)
#draw_graph(node_list, weights, "Title", "Saveas")