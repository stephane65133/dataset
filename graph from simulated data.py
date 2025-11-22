# Building an example of an intention graph from simulated data
import networkx as nx
import matplotlib.pyplot as plt

# Creating the directed graph
G = nx.DiGraph()

# Nodes: attack intentions
intentions = [
    "Reconnaissance", "Scan_HTTP", "Scan_SSH",
    "Bruteforce_SSH", "Exploitation_Vuln", "Exfiltration"
]

# Adding nodes
G.add_nodes_from(intentions)

# Edges: transitions with probabilities (from the simulated or real dataset)
transitions = [
    ("Reconnaissance", "Scan_HTTP", 0.6),
    ("Reconnaissance", "Scan_SSH", 0.4),
    ("Scan_HTTP", "Scan_SSH", 0.7),
    ("Scan_SSH", "Bruteforce_SSH", 0.8),
    ("Bruteforce_SSH", "Exploitation_Vuln", 0.5),
    ("Exploitation_Vuln", "Exfiltration", 0.9),
    ("Bruteforce_SSH", "Exfiltration", 0.3)
]

# Adding weighted edges
for u, v, w in transitions:
    G.add_edge(u, v, weight=w)

# Plotting the graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='lightblue')
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_size=9)
plt.title("Example Intention Graph Based on the Dataset")
plt.axis('off')
plt.tight_layout()
plt.show()



threshold = 0.5
G_minimized = nx.DiGraph()

# Copy nodes
G_minimized.add_nodes_from(G.nodes())

# Add only high-probability edges
for u, v, data in G.edges(data=True):
    if data['weight'] >= threshold:
        G_minimized.add_edge(u, v, weight=data['weight'])

# Plotting the minimized graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G_minimized, seed=42)
edge_labels = nx.get_edge_attributes(G_minimized, 'weight')
nx.draw_networkx_nodes(G_minimized, pos, node_size=1200, node_color='lightgreen')
nx.draw_networkx_edges(G_minimized, pos, arrowstyle='->', arrowsize=20, edge_color='black')
nx.draw_networkx_labels(G_minimized, pos, font_size=10)
nx.draw_networkx_edge_labels(G_minimized, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_size=9)
plt.title("Minimized Intention Graph (Threshold = 0.5)")
plt.axis('off')
plt.tight_layout()
plt.show()
