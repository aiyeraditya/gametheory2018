import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

def draw_graph(node_list, weights, name, saveas):
    G = nx.Graph()
    for node in node_list:
        G.add_node(node)
    pos=nx.spectral_layout(G)

    nx.draw_networkx_nodes(G,pos,node_color=node_type, node_size=750)
    for i in range(len(node_list)):
        for j in range(i+1, len((node_list))):
            G.add_edge(node_list[i], node_list[j], weight = weights[i][j])      # Define the weights

    # Draw Graph with widths corresponding to weights
    for i in range(len(node_list)):
        for j in range(len(node_list)):
            nx.draw_networkx_edges(G, pos, edgelist = [(node_list[i],node_list[j])], width = weights[i][j]/15);
                # Draws the Graph edge wise with width corresponding to weights

    arc_weight = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=arc_weight, font_color='black')
    plt.axis('off')
    plt.title(name);
    plt.savefig(saveas + ".pdf")
    #plt.show()

#Function to Delete Node
def delete(node_list, weights, node_del):
    n = [];
    w = weights;
    for node1 in node_list:
        if (node1 == -1):
            continue;
        for node2 in node_list:
            if (node2 == -1):
                continue;
            if ((node1 == node_del) or (node2 == node_del)):
                w[node1][node2] = 0;
            else:
                w[node1][node2] = weights[node1][node2];
    for node1 in node_list:
        if (node1 == node_del):
            n.append(-1);
        else:
            n.append(node1);
    return n, w


def add(nodelist, weights, weights_add):
    n1 = len(nodelist);
    nodelist.append(n1);
    print n1;
    w = [[0 for i in range(n1 + 1)] for j in range(n1 + 1)];
    for i in nodelist:
        if (i == -1):
            continue;
        for j in nodelist:
            if(j == -1):
                continue;
            if((i != n1) and (j != n1)):
                w[i][j] = weights[i][j];
            else:
                if((i == n1) and (j == n1)):
                    w[i][j] = 0;
                else:
                    if(i == n1):
                        w[i][j] = weights_add[j];
                    if(j == n1):
                        w[i][j] = weights_add[i];
    return nodelist, w;
    
def capacity(nodelist, weights, fitness, N):
    n = len(nodelist)
    if (n < N):
        return nodelist, weights, fitness;

    # Sort nodelist by fitness
    # Delete Last N-n nodes
        
        

node_list = [0,1,2]
weights = [[0,2,3],[2,0,6],[3,6,0]]
node_list, weights = add(node_list, weights, [1,2,3])
print node_list;
print weights;
