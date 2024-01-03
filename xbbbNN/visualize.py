from xbpy import rdutil, geometry
import numpy as np
from tqdm import tqdm
import pymolviz as pmv
from rdkit.Chem import rdmolops

cut_off = 5.0
sampled_max_angles = [np.pi, np.pi/8, np.pi/16]
sampled_distances = [0.4, 0.2, 0.1]
grid_density = 1.0
arrow_length = 0.5

def visualize(function, file):
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
            arrows = []
            best_scores = []
            for point in tqdm(considered_points):
                viable_acceptors = [acceptor_atoms[idx] for idx in acceptor_tree.query_ball_point([point], cut_off)[0]]
                if len(viable_acceptors) > 0:
                    next_orientation = np.array([0, 0, 1])
                    for phi, d in zip(sampled_max_angles, sampled_distances):
                        considered_orientations = geometry.sample_cone(next_orientation, 1, phi, d)
                        scores = []
                        for orientation in considered_orientations:
                            scores.append(function(point, orientation, viable_acceptors, halogen_symbol = halogen.GetSymbol(), bond_distance = neighbor_distance))
                        next_orientation = considered_orientations[np.argmax(scores)]
                    best_scores.append(max(scores))
                    arrows.append([point + (next_orientation * arrow_length), point])
            if arrows:
                print(best_scores)
                arrows = np.array(arrows)
                best_scores = np.array(best_scores)
                arrows = pmv.Arrows(arrows[best_scores > 0], color = best_scores[best_scores > 0], name = f"BP_{bp_index}_{halogen.GetSymbol()}_{halogen.GetIdx()}_Arrows", linewidth=0.05, head_length=0.6)    
                visuals[(bp_index, halogen.GetIdx())] = pmv.Group([arrows, arrows.colormap], name = f"BP_{bp_index}_{halogen.GetSymbol()}_{halogen.GetIdx()}_Group")
    return pmv.Script(visuals.values())
    
