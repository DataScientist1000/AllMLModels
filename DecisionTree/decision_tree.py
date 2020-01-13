# -*- coding: utf-8 -*-

"""
the binary tree structure;

the depth of each node and whether or not it’s a leaf;

the nodes that were reached by a sample using the decision_path method;

the leaf that was reached by a sample using the apply method;

the rules that were used to predict a sample;

the decision path shared by a group of samples.
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz
import pydotplus

import collections
import os

# Load data and store it into pandas DataFrame objects
iris = load_iris()
X = pd.DataFrame(iris.data[:,:], columns = iris.feature_names[:])
y = pd.DataFrame(iris.target, columns = ["Species"])


# Defining and fitting a DecisionTreeClassifier instance
tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X,y)

tree.tree_

"""
The decision estimator has an attribute called tree_  which stores the entire
tree structure and allows access to low level attributes. The binary tree
 tree_ is represented as a number of parallel arrays. The i-th element of each
 array holds information about the node `i`. Node 0 is the tree's root.
 
# Among those arrays, we have:
#   - left_child, id of the left child of the node
#   - right_child, id of the right child of the node
#   - feature, feature used for splitting the node
#   - threshold, threshold value at the node

# Using those arrays, we can parse the tree structure:
"""


n_nodes = tree.tree_.node_count
children_left = tree.tree_.children_left
children_right  = tree.tree_.children_right
feature = tree.tree_.feature
threshold = tree.tree_.threshold

"""
The tree structure can be traversed to compute various properties such
as the depth of each node and whether or not it is a leaf.
"""
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)

stack = [(0, -1)]  # seed is the root node id and its parent depth


while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"  % n_nodes)

for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
print()

"""
# First let's retrieve the decision path of each sample. The decision_path
# method allows to retrieve the node indicator functions. A non zero element of
# indicator matrix at the position (i, j) indicates that the sample i goes
# through the node j.
"""

node_indicator = tree.decision_path(X)

# Similarly, we can also have the leaves ids reached by each sample.
leave_id = tree.apply(X)

# Now, it's possible to get the tests that were used to predict a sample or
# a group of samples. First, let's make it for the sample.

#sample_id = 0
#node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
#                                    node_indicator.indptr[sample_id + 1]]
#
#print('Rules used to predict sample %s: ' % sample_id)
#for node_id in node_index:
#    if leave_id[sample_id] == node_id:
#        continue
#
#    if (X[sample_id, feature[node_id]] <= threshold[node_id]):
#        threshold_sign = "<="
#    else:
#        threshold_sign = ">"
#
#    print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
#          % (node_id,
#             sample_id,
#             feature[node_id],
#             X[sample_id, feature[node_id]],
#             threshold_sign,
#             threshold[node_id]))
#
## For a group of samples, we have the following common node.
#sample_ids = [0, 1]
#common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
#                len(sample_ids))
#
#common_node_id = np.arange(n_nodes)[common_nodes]
#
#print("\nThe following samples %s share the node %s in the tree"
#      % (sample_ids, common_node_id))
#print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))


# Visualize Decision Tree
export_graphviz(tree, 
                #out_file="DT_Iris.png",
                out_file=None,
                feature_names = list(X.columns),
                class_names= iris.target_names,
                filled=True,
                rounded = True       
                )


dot_data = export_graphviz(tree, 
                out_file=None,
                feature_names = list(X.columns),
                class_names= iris.target_names,
                filled=True,
                rounded = True       
                )

graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
        
graph.write_png('tree.png')
        