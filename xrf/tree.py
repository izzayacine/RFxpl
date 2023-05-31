#
#==============================================================================
from anytree import Node, RenderTree,AsciiStyle
import json
import numpy as np
import math
import os


#
#==============================================================================
class dt_node(Node):
    def __init__(self, id, parent = None):
        Node.__init__(self, id, parent)
        self.id = id  # The node value
        self.name = None
        self.left_node_id = -1   #  Left child
        self.right_node_id = -1  # Right child

        self.feature = -1
        self.threshold = None
        self.values = None
        self.label = None # np.argmax(self.values[i])
        self.proba = None

    def __str__(self):
        pref = ' ' * self.depth
        if (len(self.children) == 0):
            out = pref+ f"leaf: {self.id}  ({self.values/sum(self.values)}):c={np.argmax(self.values)}"
        else:
            if(self.name is None):
                out = pref + f"{self.id} {self.feature}<={self.threshold:.4f}"
            else:
                out = pref + f"{self.id} {self.name}<={self.threshold:.4f}"
        return out

#==============================================================================
def build_tree(tree_, feature_names = None):
    ##  
    feature = tree_.feature
    threshold = tree_.threshold
    values = tree_.value
    # 3d-array for multi-label, i.e. n_outputs_ > 1
    assert ((len(values.shape) == 3) and (values.shape[1] == 1)) 
    values = values.reshape(values.shape[0], values.shape[2]) # convert 3d-array to 2d
    normalizer = values.sum(axis=1)[:, np.newaxis]
    normalizer[normalizer == 0.0] = 1.0
    probs = values/normalizer # used for MaxSAT-encoding
    n_nodes = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaf = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
    
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaf[node_id] = True    
    ##        
    
    m = tree_.node_count  
    assert (m > 0), "Empty tree"
    
    def extract_data(idx, root = None, feature_names = None):
        i = idx
        assert (i < m), "Error index node"
        if (root is None):
            node = dt_node(i)
        else:
            node = dt_node(i, parent = root)
        #node.cover = json_node["cover"]
        if is_leaf[i]:
            node.values = values[i] # class = np.argmax(values[i])
            node.label = np.argmax(values[i])
            node.proba = probs[i]
        else:
            node.feature = feature[i]
            if (feature_names is not None):
                node.name = feature_names[feature[i]]
            node.threshold = threshold[i]
            node.left_node_id = children_left[i]
            node.right_node_id = children_right[i]
            extract_data(node.left_node_id, node, feature_names) #feat < threshold (i.e. bin '<= 0.5' False)
            extract_data(node.right_node_id, node, feature_names) #feat > threshold ( '> 0.5' True)            

        return node
    
    root = extract_data(0, None, feature_names)
    
    return root


#==============================================================================
def walk_tree(node):
    if (len(node.children) == 0):
        # leaf
        print(node)
    else:
        print(node)
        walk_tree(node.children[0])
        walk_tree(node.children[1])

def count_nodes(root):
    def count(node):
        if len(node.children):
            return sum([1+count(n) for n in node.children])
        else:
            return 0
    m = count(root) + 1
    return m

#
#==============================================================================
def predict_tree(node, sample):
    while (len(node.children) == 0):
        # leaf node
        return np.argmax(node.values)
    else:
        feature_branch = node.feature
        sample_value = sample[feature_branch]
        assert(sample_value is not None)
        if(sample_value <= node.threshold):
            return predict_tree(node.children[0], sample)
        else:
            return predict_tree(node.children[1], sample)

#def predict_tree(node, sample):
#    while (len(node.children) == 0): # not leaf node
#        feature_branch = node.feature
#        sample_value = sample[feature_branch]
#        assert(sample_value is not None)
#        if(sample_value <= node.threshold):
#            node = node.children[0]
#        else:
#            node = node.children[1]
#    return np.argmax(node.values)  

#
#==============================================================================
class Forest:
    """ An ensemble of decision trees.

    This object provides a common interface to many different types of models.
    """
    def __init__(self, ensemble, feature_names = None):
        """
            ensemble: list of (sklearn) dtrees, the model must have
            at least 1 tree
        """
        assert (len(ensemble)) 
        
        if ((feature_names is not None) and 
            (ensemble[0].n_features_in_ > len(feature_names))):
            # OHE features
            feature_names = None        
        
        self.trees = [ build_tree(dt.tree_, feature_names) for dt in ensemble]
        self.sz = sum([dt.tree_.node_count for dt in ensemble])
        self.md = max([dt.tree_.max_depth for dt in ensemble])
        self.n_classes = ensemble[0].n_classes_
        self.n_features = ensemble[0].n_features_in_
        self.attr_names = feature_names
 
        nb_nodes = [dt.tree_.node_count for dt in ensemble]
        print(f"min: {min(nb_nodes)} | max: {max(nb_nodes)}")
        assert(nb_nodes == [count_nodes(dt) for dt in self.trees])
        #self.print_trees()
        
        
    def print_trees(self):
        for i,t in enumerate(self.trees):
            print("tree number: ", i)
            walk_tree(t)
         
        
    def _apply_tree(self, root, X):
        """
            X: ndarray of shape (n_samples, n_features)
        """
        label = np.zeros((X.shape[0], self.n_classes))
        for i,data in enumerate(X):
            node = root
            while (len(node.children) > 0): # non-terminal node
                feature_branch = node.feature
                data_value = data[feature_branch]
                if(data_value <= node.threshold):
                    node = node.children[0]
                else:
                    node = node.children[1]
            label[i] = node.values    
        return label            
   
    
    def predict_proba(self, data):
        """
            apply probability argmax (sklean RF)
            data: ndarray of shape (n_features,) or (n_samples, n_features)
        """
        n_samples = data.shape[0] if len(data.shape) > 1 else 1
        if n_samples < 2: # n_samples == 1
            data = np.reshape(data, (-1, data.shape[0])) # reshape 1d-array to 2d-array 
            
        #all_proba = np.zeros((len(self.trees), n_samples, self.n_classes))
        all_proba = np.zeros((n_samples, self.n_classes))
        for t in self.trees:
            proba = self._apply_tree(t, data)
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer
            all_proba += proba
        all_proba /= len(self.trees) 
        
        y_pred = np.argmax(all_proba, axis=1) # i-th class
        # alternatively, class_names.take(np.argmax(all_proba, axis=1), axis=0)
        if n_samples == 1:
            y_pred = y_pred[0]
            
        return y_pred 
        
        
    #def predict_inst(self, data):
    #    scores = [predict_tree(dt, data) for dt in self.trees]
    #    scores = np.asarray(scores)
    #    maj = np.argmax(np.bincount(scores))
    #    return maj
    
    
    def predict(self, X):
        """
            apply majority votes (Breiman 2001)
            X: ndarray of shape (n_features,) or (n_samples, n_features)
        """
        X = np.asarray(X)
        #assert (len(X.shape) in [1,2])
        n_samples = X.shape[0] if (len(X.shape) == 2) else 1
        if n_samples < 2: # nof_samples == 1
            X = np.reshape(X, (-1, X.shape[0])) # reshape 1d to 2d
        
        votes = []
        for i in range(n_samples):
            scores = np.asarray([predict_tree(t, X[i]) for t in self.trees])
            votes.append(scores)
        votes = np.asarray(votes)    
        #np.bincount(x, weights=self._weights_not_none)
        majority = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=votes)
        if n_samples == 1:
            majority = majority[0]
        
        return majority   

