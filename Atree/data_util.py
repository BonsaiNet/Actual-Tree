import pandas as pd
import numpy as np
import scipy.io as spio

__init__= ['get_labels', 
          'location_by_vector_from_parent',
          'swc_matrix_from_cylander_model',
          'pandas_non_missing']

def get_labels(path_to_treeindex='./trees4roozbeh/treeindex.csv'):
    """
    Returning the label of each tree
    """
    df = pd.read_csv(path_to_treeindex)
    labels = []
    for i in range(100):
        index = np.where(df['random_id']==i)[0][0]
        if index<20:
            labels.append('Ghana')
        elif index<40:
            labels.append('UK')
        elif index<60:
            labels.append('Gabon')
        elif index<80:
            labels.append('Rushworth (Aus)')
        else:
            labels.append('Wytham Meteoc (UK)')
    return labels

def location_by_vector_from_parent(parent_index, vector_from_parent):
    """
    Returning the 3d location of each node, given the parent index of the node and the vector connecting 
    to the parent.
    
    Parameters:
    -----------
    parent_index : array of integers
        The index of the parent for each node of the tree. It starts from 0 and the parent
        of the root is 0.
    vector_from_parent: numpy array of size [3, number of nodes]
        For the node with index i, vector_from_parent[i, :] is the vector that connect it to its
        parent. 
    
    Returns:
    --------
    location: numpy array of size [3, number of nodes]
        The 3d location of the nodes.
    """
    up = parent_index.astype(int)
    location = np.zeros([3, len(parent_index)])
    while(sum(up) != 0):
        location += vector_from_parent[:, up]
        up = parent_index[up]
    location += vector_from_parent
    return location

def swc_matrix_from_cylander_model(cylander_model):
    """
    Making swc matrix from a cylander model (see 'What is SWC format?' in
    http://neuromorpho.org/myfaq.jsp)
    
    Parameters:
    -----------
    cylander_model: dict
        Dictionary of the all the infromation in the cylander model. The keys that are used
        in this code are: 'CPar', 'Len', 'Rad' and 'Axe'
    
    Returns:
    --------
    swc_matrix: numpy
        Swc matrix with size [number of nodes, 7]
    
    """
    n_nodes = len(cylander_model['CPar'])
    parent_index = cylander_model['CPar'] -1
    parent_index[0] = 0
    length = cylander_model['Len']
    axe = cylander_model['Axe']
    distance_from_parent = np.zeros([n_nodes, 3])
    for i in range(3):
        distance_from_parent[1:, i] = axe[1:, i] * length[:-1]
    location = location_by_vector_from_parent(parent_index,
                                              distance_from_parent.T)
    swc_matrix = np.ones([n_nodes, 7])
    swc_matrix[:, 0] = np.arange(n_nodes)+1
    swc_matrix[:, 5] = cylander_model['Rad']
    swc_matrix[:,2:5] = location.T
    swc_matrix[:,6] = cylander_model['CPar']
    swc_matrix[0,6] = -1. 
    return swc_matrix

def pandas_non_missing():
    missing = np.array([ 2,  7, 17, 19, 23, 26, 28, 34, 39, 48, 53, 54, 55, 56, 58, 59, 65, 89, 94, 99])
    list_trees = []
    labels = get_labels(path_to_treeindex='./trees4roozbeh/treeindex.csv')
    for i in range(100):
        if i not in missing:
            mat = spio.loadmat("./trees4roozbeh/"+str(i)+".mat", squeeze_me=True)
            swc_matrix = swc_matrix_from_cylander_model(mat)
            list_trees.append(swc_matrix)
        else:
            list_trees.append([])
        print(i)
    list_trees_not_missing = []
    labels_not_missing = []
    for i in range(100):
        if i not in missing:
            list_trees_not_missing.append(list_trees[i])
            labels_not_missing.append(labels[i])
    tree_data = pd.DataFrame()
    tree_data['swc'] = list_trees_not_missing
    tree_data['location'] = labels_not_missing
    return tree_data