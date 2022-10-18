import igraph as ig
import networkx as nx
from statistics import *
import math

from util_funcs import *

def k_degree(g, vertex_id):
  return g.neighborhood_size(vertex_id) - 1

def eccentricity(g, vertex_id):
  return g.eccentricity(vertex_id)

def triangles(g, vertex_id):
  triangles = g.cliques(min=3, max=3)
  count = 0
  for t in triangles:
    if vertex_id in t: count += 1
  return count

def clique_num(g, vertex_id):
  return g.clique_number()

def coreness(g, vertex_id):
  return g.coreness('all')[vertex_id]

def betweenness_centrality(g, vertex_id):
  return g.betweenness(vertex_id)

def closeness_centrality(g, vertex_id):
  return g.closeness(vertex_id)
  
def harmonic_centrality(g, vertex_id):
  return g.harmonic_centrality(vertex_id)

def eigenvector_centrality(g, vertex_id):
  # Centralities are normalized such that the maximum one is always equal to 1
  return g.eigenvector_centrality(True, True)[vertex_id]

def decay_centrality(g, vertex_id):
  distances = []
  for i in g.neighborhood(vertex_id):
    if vertex_id == i: continue
    else: distances.append(g.shortest_paths(vertex_id, i)[0][0])
  delta = 0.5 #(0<delta<1)
  return sum([delta**d for d in distances])

def pagerank(g, vertex_id):
  return g.pagerank(vertex_id)

def katz_centrality(g, vertex_id):
    g_nx = ig_to_nx(g)
    return nx.katz_centrality(g_nx, alpha=0.1, beta=1.0)[vertex_id]

def local_clustering_coeff(g, vertex_id):
  return g.transitivity_local_undirected(mode='zero')[vertex_id]
  
def global_clustering_coeff(g, vertex_id):
  return g.transitivity_undirected(mode='zero')

def shannon_diversity(g, vertex_id):
  return g.diversity(vertex_id)

def h_index_helper(iterable, sorted=False):
    if not isinstance(iterable, list):
        iterable = list(iterable)
    if not sorted:
        iterable.sort()

    value_count = len(iterable)
    for index, value in enumerate(iterable):
        result = value_count - index
        if result <= value:
            return result
    return 0

def h_index(g, vertex_id):
    return None

def neighborhood_density(g, vertex_id):
    return g.density()

def rwr(g, vertex_id):
  reset_vector = [0]*total_vertices(g)
  reset_vector[vertex_id] = 1
  return g.personalized_pagerank(vertices=vertex_id,  reset=reset_vector)

def lpi(g, vertex_id):
  return None

def lrw(g, vertex_id):
  return g.personalized_pagerank(vertices=vertex_id)






