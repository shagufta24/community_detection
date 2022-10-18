import igraph as ig
import networkx as nx
from statistics import *
import math 

def ig_to_nx(g):
  edge_list = g.get_edgelist()
  return nx.Graph(edge_list)

def list_diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

# Some standard values for reuse
def degree_list(g):
    return g.degree()

def adjacency_list(g):
    return list(map(set, g.get_adjlist()))

def max_degree(g):
    return max(degree_list(g))

def min_degree(g):
    min(degree_list(g))

def total_vertices(g):
    return len(degree_list(g))

# List of indices of vertices of a graph
def vertex_list(g):
  v_list = [v.index for v in g.vs()]
  return v_list

# List of vertices that have a particular degree d
def vertices_of_degree(g, d):
  degree_list = g.degree()
  v_list = [i for i in range(len(degree_list)) if degree_list[i] == d]
  return v_list

# To find induced subgraph at a given level
def level_induced_subgraph(g, vertex_id, level):
    neighbors = g.neighborhood(vertex_id, level)
    g_induced = g.induced_subgraph(neighbors)

    # Mapping name attribute of original graph nodes to induced subgraph nodes
    seq = g.vs.select(neighbors)
    name_list = [v["name"] for v in seq]
    g_induced.vs["name"] = name_list
    return g_induced

# To find the new vertex id in subgraph using name attribute
def reset_id(g, vertex_id):
  for v in g.vs():
      if v["name"] == vertex_id:
        vertex_id_new = v.index
  return vertex_id_new

def degree_dist(g, vertex_id, level):
  # Getting neighborhood of given level (node itself not included) 
  neighbors = g.neighborhood(vertex_id, level)

  # For each degree, finding the number of neighbors
  frequency = []
  for deg in range(0, max_degree(g) + 1):
    count = 0
    for n in neighbors:
      if (degree_list(g)[n] == deg):
        count += 1
    frequency.append(count)
  return frequency

# Some functions to apply on the degree distribution
def dist_max(dist):
  return max(dist)

def dist_min(dist):
  return min(dist)

def dist_mean(dist):
  return mean(dist)

def dist_median(dist):
  return median(dist)

def dist_mode(dist):
  return mode(dist)

def dist_geometric_mean(dist):
  return geometric_mean(dist)

def dist_harmonic_mean(dist):
  return harmonic_mean(dist)

def dist_stdev(dist):
  return stdev(dist)

def dist_pstdev(dist):
  return pstdev(dist)

def dist_variance(dist):
  return variance(dist)

def dist_pvariance(dist):
  return pvariance(dist)
