import pandas as pd
import statistics
import math
import numpy as np

df = pd.read_csv('features_data.csv')
comms = pd.read_csv('comms_data.csv')

# Drop extra columns
df.drop(columns = df.columns[[0,1]], axis=1, inplace=True)

# Temporarily dropping
df.drop(['shannon_node_1', 'shannon_node_2'], axis=1, inplace=True)

# Creating product features
new_col_names = ['degree', 'triangles', 'clique_number', 'coreness', 'betweenness', 'closeness', 'harmonic', 'eigenvector' , 'decay', 'pagerank', 'local_clustering_coeff', 'global_clustering_coeff', 'h_index', 'nbd_density', 'RWR', 'LRW']
df2 = pd.DataFrame()
for i in range(32):
    df2[new_col_names[int(i/2)]] = df[df.columns[i]] * df[df.columns[i+1]]
    i += 2
for i in range(32, 50):
    col_name = df.columns[i]
    df2[col_name] = df.iloc[:, i]
df = df2

# Random oversampling
# class count
class_count_0, class_count_1 = df['label'].value_counts()
# separate classes
class_0 = df[df['label'] == 0]
class_1 = df[df['label'] == 1]
      
class_1_over = class_1.sample(class_count_0, replace=True)
test_over = pd.concat([class_1_over, class_0], axis=0)

print(test_over['label'].value_counts())

# plot the count
print(test_over['label'].value_counts().plot(kind='bar', title='count (target)'))

