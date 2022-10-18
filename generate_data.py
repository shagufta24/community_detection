import igraph as ig
import networkx as nx
import pandas as pd
from tqdm import tqdm

from util_funcs import *
from single_funcs import *
from pair_funcs import *

# Data file path
graph_name = "power"
gml_file_path = "datasets/" + graph_name + ".gml"
output_path = graph_name + "_data"

# Read GML file to Igraph network
try:
    graph_ig = ig.Graph.Read_GML(gml_file_path)
except OSError as e:
    print("File not found.")
    quit()

print("Graph read successfully.")

# Converting igraph to networkx
graph_nx = ig_to_nx(graph_ig)

# Assigning a name attribute to each vertex, whose value is the same as the vertex index
graph_ig.vs["name"] = [v.index for v in graph_ig.vs()]

# Listing all functions
single_funcs = [k_degree, eccentricity, triangles, clique_num, coreness, betweenness_centrality, closeness_centrality, harmonic_centrality, eigenvector_centrality, decay_centrality, pagerank, katz_centrality, local_clustering_coeff, global_clustering_coeff, shannon_diversity, h_index, neighborhood_density, rwr, lpi, lrw]
# pair_funcs = [are_connected, path_exists, total_simple_paths, shortest_path, common_triangles, common_neighbors, cosine_similarity, pearson_similarity, euclidean_distance, manhattan_distance, hamming_distance, covariance, rand_index, jaccard_index, sorenson_dice, hub_promoted, hub_depressed, lhn, res_alloc, adamic_adar, pref_attach, overlap_coeff, nbd_overlap, edge_conn, vertex_conn, katz_index, hitting_time, avg_commute_time, cosine_l, l_plus, mfi]
pair_funcs = [are_connected, common_triangles, common_neighbors, cosine_similarity, pearson_similarity, euclidean_distance, manhattan_distance, hamming_distance, covariance, rand_index, jaccard_index, sorenson_dice, hub_promoted, hub_depressed, lhn, res_alloc, adamic_adar, pref_attach, overlap_coeff, nbd_overlap, edge_conn, vertex_conn, katz_index, avg_commute_time, cosine_l, l_plus, mfi]

# Create dataframe to store function outputs
total_vertices = total_vertices(graph_ig)
row_count = int(total_vertices * (total_vertices-1) / 2)
node_1 = []
node_2 = []
for i in range(total_vertices):
  for j in range(i, total_vertices):
    if i!=j:
      node_1.append(i)
      node_2.append(j)

df = pd.DataFrame()
df['node_1'] = node_1
df['node_2'] = node_2

print("Computing subgraphs...")

level = 1 # Keeping it 1 for now
# Computing subgraphs of all nodes
subgraphs_ig = []
subgraphs_nx = []
for vertex_id in range(total_vertices):
    induced_ig = level_induced_subgraph(graph_ig, vertex_id, level)
    subgraphs_ig.append(induced_ig)
    induced_edge_list = induced_ig.get_edgelist()
    induced_nx = nx.Graph(induced_edge_list)
    subgraphs_nx.append(induced_nx)

print("Computing single node features...")

# Single node features
for func in tqdm(single_funcs):
    print(func)
    node_1 = []
    node_2 = []
    for i in range(row_count):
        index_1 = int(df['node_1'][i])
        index_2 = int(df['node_2'][i])
        subgraph_1 = subgraphs_ig[index_1]
        subgraph_2 = subgraphs_ig[index_2]
        new_index_1 = reset_id(subgraph_1, index_1)
        new_index_2 = reset_id(subgraph_2, index_2)
        node_1.append(func(subgraph_1, new_index_1))
        node_2.append(func(subgraph_2, new_index_2))

    df[func.__name__ + '_1'] = node_1
    df[func.__name__ + '_2'] = node_2

print("Computing node pair features...")

# Pair node features
for func in tqdm(pair_funcs):
    result = []
    for i in range(row_count):
        index_1 = int(df['node_1'][i])
        index_2 = int(df['node_2'][i])
        result.append(func(graph_ig, index_1, index_2))
    df[func.__name__] = result

df.to_csv(output_path + 'features.csv')

print("Features obtained successfully.")

# Remove Nan columns
df = df.dropna(axis=1)
df = df.reset_index(drop=True)

# Read communities data

comms_data = pd.read_csv(output_path + 'comms_data.csv')
comms_data = comms_data.iloc[:, 1:]

print("Community data found.")

# Compressing comms data into one columns (labels)
comms_data['label'] = comms_data.iloc[:, 2:].mean(axis=1)
comms_data['label'] = comms_data['label'].apply(lambda x: 0 if (x < 0.5) else 1)

# Add labels to df -> dataset created
df['label'] = comms_data['label']

# Save dataset
df.to_csv(output_path + 'dataset.csv')

print("Dataset created successfully.")

