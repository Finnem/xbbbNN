from xbpy import rdutil, geometry
import numpy as np
from tqdm import tqdm
import pymolviz as pmv
from rdkit.Chem import rdmolops
from scipy.spatial import cKDTree
from .feature_extraction.bb_features import get_bb_head_on_features_from_acceptor_oxygen

cut_off = 6.0
sampled_max_angles = [np.pi, np.pi/8, np.pi/16]
sampled_distances = [0.4, 0.2, 0.1]
grid_density = 1.0
arrow_length = 0.5
score_filter_threshold = -5e-2

def visualize_pockets(function, file, with_pseudo_atoms = False):
    protein = rdutil.read_molecules(file, proximityBonding=True)[0]
    #protein = rdmolops.AddHs(protein)
    binding_pockets = rdutil.get_binding_pockets_by_ligand(protein)

    visuals = {}
    for bp_index, binding_pocket in enumerate(binding_pockets):
        acceptor_atoms = np.array([atom for atom in binding_pocket.pocket_atoms if atom.GetSymbol() == "O"])
        #return binding_pocket.pocket_atoms
        #return acceptor_atoms
        acceptor_tree = rdutil.geometry.AtomKDTree(acceptor_atoms)
        # for now we lazily assume every oxygen to be a treatable acceptor, the functions have to deal with that

        considered_points = binding_pocket.get_accessible_points(grid_density)

        relevant_halogens = []
        for ligand in binding_pocket.ligands:
            relevant_halogens += [atom for atom in ligand if atom.GetSymbol() in ["Cl", "Br", "I"]]
            neighbor_distances = np.array([np.linalg.norm(rdutil.position(x) - rdutil.position(x.GetNeighbors()[0])) for x in relevant_halogens])
        for halogen, neighbor_distance in zip(relevant_halogens, neighbor_distances):
            best_orientations, best_scores = get_best_orientations(considered_points, function, halogen.GetSymbol(), neighbor_distance, acceptor_atoms, acceptor_tree)
            arrows = np.array(best_orientations)
            best_scores = np.array(best_scores)
            score_filter = best_scores < score_filter_threshold
            pseudo_atoms = pmv.PseudoAtoms(arrows[:,0][score_filter], best_scores[score_filter], name=f"BP_{bp_index}_{halogen.GetSymbol()}_{halogen.GetIdx()}_PseudoAtoms")
            arrows = pmv.Arrows(arrows[score_filter], color = best_scores[score_filter], name = f"BP_{bp_index}_{halogen.GetSymbol()}_{halogen.GetIdx()}_Arrows", linewidth=0.05, head_length=0.6)    
            if with_pseudo_atoms:
                visuals[(bp_index, halogen.GetIdx())] = pmv.Group([arrows, arrows.colormap, pseudo_atoms], name = f"BP_{bp_index}_{halogen.GetSymbol()}_{halogen.GetIdx()}_Group")
            else:
                visuals[(bp_index, halogen.GetIdx())] = pmv.Group([arrows, arrows.colormap], name = f"BP_{bp_index}_{halogen.GetSymbol()}_{halogen.GetIdx()}_Group")
    return pmv.Script(visuals.values())




def get_best_bb_side_on_orientations(acceptors, positions, halogen_symbol, bond_distance, model):
    acceptors = np.array(acceptors)
    position_kd_tree = cKDTree(positions)
    acceptor_positions = np.array([rdutil.position(acceptor) for acceptor in acceptors])
    position_filters = {}
    
    # initial filter for unnecessary positions
    acceptor_filter = np.ones(len(acceptors), dtype = bool)
    for i, query_result in enumerate(position_kd_tree.query_ball_point(acceptor_positions, cut_off)):
        if len(query_result) == 0:
            acceptor_filter[i] = False

        position_filters[i] = query_result

    # sampling all orientations:
    best_orientations = np.full(positions.shape, [1,0,0])
    for phi, d in zip(sampled_max_angles, sampled_distances):
        # intializing orientation_values
        possible_orientations = geometry.sample_cone(best_orientations, 1, phi, d)
        orientation_values = np.zeros(possible_orientations.shape[:-1])
        for i, acceptor in zip(np.arange(len(acceptors))[acceptor_filter], acceptors[acceptor_filter]):
            position_filter = position_filters[i]
            filtered_orientations = possible_orientations[position_filter]; filtered_positions = positions[position_filter]
            acceptor_position = rdutil.position(acceptor)
            broadcasted_positions = np.broadcast_to(filtered_positions[:,None,:], filtered_orientations.shape)
            neighbor_positions = broadcasted_positions - (bond_distance * filtered_orientations)
            features = get_bb_side_on_features_from_acceptor_oxygen(acceptor, broadcasted_positions.reshape(-1, 3), neighbor_positions.reshape(-1, 3), halogen_symbol)
            orientation_values[position_filter] += model(features).reshape(orientation_values[position_filter].shape)
            #print(orientation_values[position_filter])
        best_indices = np.argmin(orientation_values.reshape(len(positions), -1), axis = 1)
        best_orientation_values = orientation_values.reshape(len(positions), -1)[np.arange(len(positions)), best_indices]
        best_orientations = possible_orientations[np.arange(len(positions)), best_indices]
    best_neighbor_positions = positions - (bond_distance * .3 * best_orientations)
    return positions, best_neighbor_positions, best_orientation_values


def get_best_bb_head_on_orientations(acceptors, positions, halogen_symbol, bond_distance, model):
    acceptors = np.array(acceptors)
    position_kd_tree = cKDTree(positions)
    acceptor_positions = np.array([rdutil.position(acceptor) for acceptor in acceptors])
    position_filters = {}
    
    # initial filter for unnecessary positions
    acceptor_filter = np.ones(len(acceptors), dtype = bool)
    for i, query_result in enumerate(position_kd_tree.query_ball_point(acceptor_positions, cut_off)):
        if len(query_result) == 0:
            acceptor_filter[i] = False

        position_filters[i] = query_result

    # sampling all orientations:
    best_orientations = np.full(positions.shape, [1,0,0])
    for phi, d in zip(sampled_max_angles, sampled_distances):
        # intializing orientation_values
        possible_orientations = geometry.sample_cone(best_orientations, 1, phi, d)
        orientation_values = np.zeros(possible_orientations.shape[:-1])
        for i, acceptor in zip(np.arange(len(acceptors))[acceptor_filter], acceptors[acceptor_filter]):
            position_filter = position_filters[i]
            filtered_orientations = possible_orientations[position_filter]; filtered_positions = positions[position_filter]
            acceptor_position = rdutil.position(acceptor)
            broadcasted_positions = np.broadcast_to(filtered_positions[:,None,:], filtered_orientations.shape)
            neighbor_positions = broadcasted_positions - (bond_distance * filtered_orientations)
            features = get_bb_head_on_features_from_acceptor_oxygen(acceptor, broadcasted_positions.reshape(-1, 3), neighbor_positions.reshape(-1, 3), halogen_symbol)
            orientation_values[position_filter] += model(features).reshape(orientation_values[position_filter].shape)
            #print(orientation_values[position_filter])
        best_indices = np.argmin(orientation_values.reshape(len(positions), -1), axis = 1)
        best_orientation_values = orientation_values.reshape(len(positions), -1)[np.arange(len(positions)), best_indices]
        best_orientations = possible_orientations[np.arange(len(positions)), best_indices]
    best_neighbor_positions = positions - (bond_distance * .3 * best_orientations)
    return positions, best_neighbor_positions, best_orientation_values
    











def get_best_orientations(considered_points, function, halogen_symbol, neighbor_distance, acceptor_atoms, acceptor_tree = None):
    arrows = []
    best_scores = []

    for point in tqdm(considered_points):
        if acceptor_tree is None:
            viable_acceptors = [atom for atom in acceptor_atoms if np.linalg.norm(rdutil.position(atom) - point) < cut_off]
        else:
            viable_acceptors = [acceptor_atoms[idx] for idx in acceptor_tree.query_ball_point([point], cut_off)[0]]
        if len(viable_acceptors) > 0:
            next_orientation = rdutil.position(viable_acceptors[0]) - point; next_orientation /= np.linalg.norm(next_orientation)
            for phi, d in zip(sampled_max_angles, sampled_distances):
                considered_orientations = geometry.sample_cone(next_orientation, 1, phi, d)
                scores = function(point, considered_orientations, viable_acceptors, halogen_symbol = halogen_symbol, bond_distance = neighbor_distance)
                next_orientation = considered_orientations[np.argmin(scores)]
            best_scores.append(min(scores))
            arrows.append([point + (next_orientation * arrow_length), point])
    return arrows, best_scores
        