import igraph as ig
import networkx as nx
from statistics import *
import math 
import numpy as np
from math import sqrt
from scipy import stats
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import adjusted_rand_score

from util_funcs import *

def are_connected(g, v1, v2, level = 1):
  if (g.are_connected(v1, v2) == True): return 1
  else: return 0

def path_exists(g, v1, v2, level = 1):
  g_nx = ig_to_nx(g)
  paths = [path for path in nx.all_simple_paths(g_nx, source = v1, target = v2)]
  if len(paths) >= 1: return 1
  return 0

def total_simple_paths(g, v1, v2, level = 1):
  g_nx = ig_to_nx(g)
  paths = nx.all_simple_paths(g_nx, source = v1, target = v2)
  return len(list(paths))

def shortest_path(g, v1, v2, level = 1):
  return g.shortest_paths(v1, v2)[0][0]

def common_triangles(g, v1, v2, level = 1):
  cliques = g.cliques(min=3,max=3)
  v1_cliques = []
  v2_cliques = []
  for t in cliques:
    if v1 in t: v1_cliques.append(t)
    if v2 in t: v2_cliques.append(t)
  common_cliques = list(set(v1_cliques) & set(v2_cliques))
  return len(common_cliques)

def common_neighbors(g, v1, v2, level = 1):
  nh1 = g.neighborhood(v1, level)
  nh2 = g.neighborhood(v2, level)
  common_neighbors = list(set(nh1) & set(nh2))
  return len(common_neighbors)

def cosine_similarity(g, v1, v2, level=1, func=degree_dist):
  a = func(g, v1, level)
  b = func(g, v2, level)
  cos_sim = dot(a, b)/(norm(a)*norm(b))
  return cos_sim

def pearson_similarity(g, v1, v2, level=1, func=degree_dist):
  a = func(g, v1, level)
  b = func(g, v2, level)
  return stats.pearsonr(a, b)[0]

def euclidean_distance(g, v1, v2, level=1, func=degree_dist):
  a = func(g, v1, level)
  b = func(g, v2, level)
  return sqrt(sum((e1-e2)**2 for e1, e2 in zip(a,b)))

def manhattan_distance(g, v1, v2, level=1, func=degree_dist):
  a = func(g, v1, level)
  b = func(g, v2, level) 
  return sum(abs(e1-e2) for e1, e2 in zip(a,b))

def hamming_distance(g, v1, v2, level=1, func=degree_dist):
  a = func(g, v1, level)
  b = func(g, v2, level)
  return (sum(abs(e1 - e2) for e1, e2 in zip(a,b))/len(a))

def covariance(g, v1, v2, level=1, func=degree_dist):
    return None

def rand_index(g, v1, v2, level=1, func=degree_dist):
  a = func(g, v1, level)
  b = func(g, v2, level)
  return adjusted_rand_score(a, b)

def jaccard_index(g, v1, v2, level=1):
  return g.similarity_jaccard(pairs = [(v1, v2)])[0]

def sorenson_dice(g, v1, v2, level=1):
  return g.similarity_dice(pairs = [(v1, v2)])[0]
  
def hub_promoted(g, v1, v2, level=1):
  intersection = len(adjacency_list(g)[v1].intersection(adjacency_list(g)[v2]))
  min_degree = min(degree_list(g)[v1], degree_list(g)[v2])
  return intersection/min_degree

def hub_depressed(g, v1, v2, level=1):
  intersection = len(adjacency_list(g)[v1].intersection(adjacency_list(g)[v2]))
  max_degree = max(degree_list(g)[v1], degree_list(g)[v2])
  return intersection/max_degree

def lhn(g, v1, v2, level=1):
  list1 = list_diff(g.neighborhood(v1, level), g.neighborhood(v1, 0))
  list2 = list_diff(g.neighborhood(v2, level), g.neighborhood(v2, 0))
  intersection = len(list(set(list1).intersection(list2)))
  degree_product = (len(list1)) * (len(list2))
  return intersection/degree_product

def res_alloc(g, u, v, level=1):
  g_nx = ig_to_nx(g)
  preds = nx.resource_allocation_index(g_nx, [(u, v)])
  for u, v, p in preds:
    return p

def adamic_adar(g, v1, v2, level=1):
  return g.similarity_inverse_log_weighted(vertices = [v1])[0][v2]

def pref_attach(g, v1, v2, level=1):
  return degree_list(g)[v1] * degree_list(g)[v2]

def overlap_coeff(g, v1, v2, level=1):
  intersection = len(set(g.neighbors(v1, level)).intersection(g.neighbors(v2, level)))
  min_nbd = len(min(g.neighbors(v1, level), g.neighbors(v2, level)))
  return intersection/min_nbd

def nbd_overlap(g, v1, v2, level=1):
    nh1 = g.neighborhood(v1, level)
    nh2 = g.neighborhood(v2, level)
    inter_list = list(set.intersection(set(nh1), set(nh2)))
    intersection_len = len(inter_list)
    union = g.neighborhood_size(v1, level) + g.neighborhood_size(v2, level) - intersection_len - 2
    return intersection_len/union

def edge_conn(g, v1, v2, level=1):
  nh1 = g.neighborhood(v1, level)
  g1 = level_induced_subgraph(g, v1, level)
  if v2 not in nh1:
    ec1 = g1.ecount()
  else:
    v1_new = reset_id(g1, v1)
    v2_new = reset_id(g1, v2)
    ec1 = g1.edge_connectivity(v1_new, v2_new)

  nh2 = g.neighborhood(v2, level)
  g2 = level_induced_subgraph(g, v2, level)
  if v1 not in nh2:
    ec2 = g2.ecount()
  else:
    v1_new = reset_id(g2, v1)
    v2_new = reset_id(g2, v2)
    ec2 = g2.edge_connectivity(v1_new, v2_new)

  return min(ec1, ec2)

def vertex_conn(g, v1, v2, level=1):
    return None

def katz_index(g, v1, v2, level=1):
    return None

def hitting_time(g, v1, v2, level=1):
  hit_idx = (v1, v2)

  A = g.get_adjacency() 
  A = np.array(A.data)
  A[hit_idx[1],:] = 0
  A[hit_idx[1],hit_idx[1]] = 1
  A = (A.T/A.sum(axis=1)).T

  B = A.copy()
  Z = []
  while(True):
      Z.append( B[hit_idx] )
      if B[hit_idx] > 0.99999999999999:
        break
      B = dot(B,A)
  return len(Z)

def laplacian(g):
    return g.laplacian()

def avg_commute_time(g, v1, v2, level=1):
  M = g.ecount() 
  L = laplacian(g)
  L_pinv = np.linalg.pinv(L) # Pseudoinverse of L
  act = M*(L_pinv[v1][v1]+L_pinv[v2][v2]-(2*L_pinv[v1][v2])) # ACT formula
  act_similarity = 1/act # Taking the inverse, since smaller ACT means more similarity
  return act_similarity

def cosine_l(g, v1, v2, level = 1):
  L = laplacian(g)
  L_pinv = np.linalg.pinv(L) # Pseudoinverse of L
  cosine_L = L_pinv[v1][v2] / sqrt(dot(L_pinv[v1][v1], L_pinv[v2][v2]))
  return cosine_L

def l_plus(g, v1, v2, level = 1):
  L = laplacian(g)
  L_pinv = np.linalg.pinv(L)
  return L_pinv[v1][v2]

def mfi(g, v1, v2, level = 1):
  A = g.get_adjacency() 
  A = np.array(A.data)
  I = np.identity(len(A), dtype = None) 
  L = laplacian(g)
  S = np.linalg.pinv(I + L)
  return S[v1][v2]





