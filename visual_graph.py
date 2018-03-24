import matplotlib.pyplot as plt
import networkx as nx

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
            G.add_edge(node_list[i], node_list[j], weight = weights[i][j])
            
    # Draw Graph with widths corresponding to weights
    for i in range(len(node_list)):
        for j in range(len(node_list)):
            nx.draw_networkx_edges(G, pos, edgelist = [(node_list[i],node_list[j])], width = weights[i][j]/15);
    plt.axis('off')
    plt.title(name);
    plt.savefig(saveas + ".pdf")
    plt.show()

# Function For Runing The Game
def evolve(node_list, weights, cycle):
    for i in range(cycles):
        print("Evolution Function Yet To be Made");
        # Rule for Evolution Here




node_list = [1,2,3]
weights = [[0, 1, 2], [1, 0, 3], [2,3,0]]
evolve(node_list, weights, cycle)
#draw_graph(node_list, weights, "Title", "Saveas")


