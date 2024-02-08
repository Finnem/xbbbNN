import numpy as np

def _get_paired_vectors(varying_coords, fixed_coords, pair_indices):
    """ Returns distances between varying coords and fixed_coords as indexed by pair_indices.

    Parameters
    ----------
    varying_coords : np.ndarray
        Coordinates of varying atoms. Should be of shape (n_variations, n_varying_atoms, 3)
    fixed_coords : np.ndarray
        Coordinates of fixed atoms. Should be of shape (n_fixed_atoms, 3)
    pair_indices : np.ndarray
        Indices of the pairs to be computed. Should be of shape (n_pairs, 2), with varying indices in the first column and fixed indices in the second.

    Returns
    -------
    np.ndarray
        The vectors between varying_coords and fixed_coords as indexed by pair_indices.
    """
    return varying_coords[:,pair_indices[:,0],:] - fixed_coords[None,pair_indices[:,1],:]



def _get_reindexing(array, filter = None):
    '''
    Computes the reindexing of an array, such that the array is sorted along the last axis at the filtered positions.

    Parameters
    ----------

    array (np.ndarray): The array to be reindexed.
    filter (np.ndarray): The filter to be applied to the array. If None, the array is not filtered.

    Returns
    -------
    np.ndarray: The reindexing of the array. This should then be applied using np.take_along_axis.
    '''
    if filter is None:
        sub_array = array
    else:
        sub_array = array[:,:,filter]
    
    sub_reindexing = np.argsort(sub_array, axis = -1)
    full_reindexing = np.full(array.shape, np.arange(array.shape[-1]), dtype = int)
    full_reindexing[:,:,filter] = np.take_along_axis(full_reindexing[:,:,filter], sub_reindexing, axis = -1)
    return full_reindexing

def get_distances_and_angles(fixed_coords, varying_coords, distance_indices, angle_indices, equivalent_indices = None):
    '''
    Computes the distances and cosine similarity for a given set of coordinates. Indexing starts at fixed_coords and continues through varying.
    I.e. to to reference the i'th fixed coord use i, but to reference the i'th varying coord, use len(fixed_coords) + i.

    Parameters
    ----------
    fixed_coords (np.ndarray): The coordinates of the fixed atoms. Should be of shape (n_fixed_atoms, 3).
    varying_coords (np.ndarray): The coordinates of the varying atoms. Should be of shape (n_variations, n_varying_atoms, 3).
    distance_indices (np.ndarray): The indices of the pairwise distances to be computed. Should be of shape (n_distances, 2).
    angle_indices (np.ndarray): The indices of the angles to be computed between fixed and varying. Should be of shape (n_angles, 4).
    equivalent_indices (np.ndarray): The indices of the equivalent atoms. Should be of shape (n_equivalent_sets, n_equivalent_atoms).
        For each equivalent set, the result will be reordered such that the closest atom is the one with the lowest index.

    Returns
    -------
    np.ndarray: The distances and cosine similarities between the given coordinates. Should be of shape (n_variations, n_distances + n_angles).
    '''

    if len(varying_coords.shape) == 2:
        varying_coords = varying_coords[None,:,:]
    if len(fixed_coords.shape) == 1:
        fixed_coords = fixed_coords[None,:]
    if equivalent_indices is None:
        equivalent_indices = []


    # compute pairwise distances, in order to avoid redundant computation of distances, we sort and eliminate duplicates from the indices
    distance_indices = np.reshape(distance_indices, (-1,2))
    angle_indices = np.reshape(angle_indices, (-1,2))
    all_indices = np.vstack((distance_indices, angle_indices))
    unique_index_pairs, invert_unique = np.unique(all_indices, axis = 0, return_inverse = True)
    
    # identify fixed-fixed, fixed-varying and varying-varying distances
    fixed_fixed_indices = (unique_index_pairs[:,0] < len(fixed_coords)) & (unique_index_pairs[:,1] < len(fixed_coords))
    fixed_varying_indices = (unique_index_pairs[:,0] < len(fixed_coords)) & (unique_index_pairs[:,1] >= len(fixed_coords))
    varying_fixed_indices = (unique_index_pairs[:,0] >= len(fixed_coords)) & (unique_index_pairs[:,1] < len(fixed_coords))
    varying_varying_indices = (unique_index_pairs[:,0] >= len(fixed_coords)) & (unique_index_pairs[:,1] >= len(fixed_coords))
    

    #  compute fixed-fixed distances
    fixed_fixed_differences = fixed_coords[unique_index_pairs[fixed_fixed_indices,0],:] - fixed_coords[unique_index_pairs[fixed_fixed_indices,1],:]
    fixed_fixed_distances = np.linalg.norm(fixed_fixed_differences, axis = -1)
    

    # compute fixed-varying distances
    used_pairs = np.vstack([unique_index_pairs[varying_fixed_indices,:],\
                            np.dstack([unique_index_pairs[fixed_varying_indices,1], unique_index_pairs[fixed_varying_indices,0]]).squeeze()]) # invert fixed-varying pairs
    used_pairs[:,0] -= len(fixed_coords) # remove fixed offset of varying indices

    both_differences = _get_paired_vectors(varying_coords, fixed_coords, used_pairs)
    varying_fixed_differences = both_differences[:,len(varying_fixed_indices):,:] # split back into varying-fixed and fixed-varying differences
    fixed_varying_differences = -both_differences[:,:len(varying_fixed_indices),:] # reinvert fixed-varying differences
    varying_fixed_distances = np.linalg.norm(varying_fixed_differences, axis = -1)
    fixed_varying_distances = np.linalg.norm(fixed_varying_differences, axis = -1)

    #compute varying-varying distances
    varying_varying_differences = varying_coords[:,unique_index_pairs[varying_varying_indices,0] - len(fixed_coords),:] - varying_coords[:,unique_index_pairs[varying_varying_indices,1] - len(fixed_coords),:]
    varying_varying_distances = np.linalg.norm(varying_varying_differences, axis = -1)

    # reconstruct requested distance order
    unique_differences = np.zeros((varying_coords.shape[0], unique_index_pairs.shape[0], 3), dtype = float)
    unique_differences[:, fixed_fixed_indices] = fixed_fixed_differences[None, :]
    unique_differences[:, fixed_varying_indices] = fixed_varying_differences
    unique_differences[:, varying_fixed_indices] = varying_fixed_differences
    unique_differences[:, varying_varying_indices] = varying_varying_differences
    unique_distances = np.linalg.norm(unique_differences, axis = -1)

    # determine equivalent differences
    class_start = len(fixed_coords) + len(varying_coords)
    index_equivalence_classes = {}
    for i, indices in zip(range(class_start, class_start + len(equivalent_indices)), equivalent_indices):
        for index in indices:
            index_equivalence_classes[index] = i
    equivalent_differences = {}
    for i, index_pair in enumerate(unique_index_pairs):
        equi_class_0, equi_class_1 = index_equivalence_classes.get(index_pair[0], index_pair[0]), index_equivalence_classes.get(index_pair[1], index_pair[1])
        if (equi_class_0 >= class_start) or (equi_class_1 >= class_start):
            equivalence_class = (equi_class_0, equi_class_1)
            current_equivalence_pair_idxs = equivalent_differences.get(equivalence_class, [])
            current_equivalence_pair_idxs.append(i)
            equivalent_differences[equivalence_class] = current_equivalence_pair_idxs

    # sort equivalent differences
    for equivalence_class, equivalence_pair_idxs in equivalent_differences.items():
        equivalence_pair_idxs = np.array(equivalence_pair_idxs)
        # indices get sorted first
        equivalence_pair_idxs = equivalence_pair_idxs[np.lexsort(unique_index_pairs[equivalence_pair_idxs].T)]
        # then distances get sorted
        equivalence_pair_distances = unique_distances[:, equivalence_pair_idxs]
        order = np.argsort(equivalence_pair_distances, axis = -1)
        new_equivalence_pair_idxs = equivalence_pair_idxs[order]
        unique_distances[:,equivalence_pair_idxs] = np.take_along_axis(unique_distances, new_equivalence_pair_idxs, axis = -1)
        unique_differences[:,equivalence_pair_idxs] = np.take_along_axis(unique_differences, new_equivalence_pair_idxs[:, :, None], axis = 1)

    distance_features = unique_distances[:,invert_unique[:distance_indices.shape[0]]]
    
    
    # compute requested angles
    unique_angle_indices = invert_unique[distance_indices.shape[0]:].reshape(-1,2)
    angle_features = np.sum(unique_differences[:,unique_angle_indices[:,0]] * unique_differences[:,unique_angle_indices[:,1]], axis = -1) / (unique_distances[:,unique_angle_indices[:,0]] * unique_distances[:,unique_angle_indices[:,1]])
    return distance_features, angle_features


