from .core import *
import numpy as np
from xbpy import rdutil
from .util import _infer_single_hydrogen_position

# =============================================================================
# bb_head_on_model
# =============================================================================

def get_bb_head_on_model_features(F_O_coord, F_C_coord, F_N_coord, F_p_NDir_coord, L_X_coords, L_C1_coords, halogen_type):
    """ Computes features for the bb_head_on_model from the given coordinates.

    Parameters
    ----------
    F_O_coord : np.ndarray
        Coordinates of the O atom of the fixed group. Should be of shape (3,).
    F_C_coord : np.ndarray
        Coordinates of the C atom of the fixed group. Should be of shape (3,).
    F_N_coord : np.ndarray
        Coordinates of the N atom of the fixed group. Should be of shape (3,).
    F_p_NDir_coord : np.ndarray
        Coordinates of the Pseudoatom given by the mean position of the heavy atoms connected to the N atom of the fixed group. Should be of shape (3,).
    L_X_coords : np.ndarray
        Coordinates of the X atom of the ligand. Should be of shape (n_variations, 3).
    L_C1_coords : np.ndarray
        Coordinates of the C1 atom of the ligand. Should be of shape (n_variations, 3).
    halogen_type : str
        The type of halogen. Should be one of "Cl", "Br" or "I".

    Returns
    -------
    np.ndarray
        The computed features. Should be of shape (n_variations, 9).
    """
    # distance features:
    F_O_idx, F_C_idx, F_N_idx, F_p_NDir_idx, L_X_idx, L_C1_idx = np.arange(6)
    distance_features = ((F_O_idx, L_X_idx), (F_N_idx, L_X_idx))
    distance_features = ((F_O_idx, L_X_idx), (F_N_idx, L_X_idx))
    angle_features = (
        ((F_O_idx, F_C_idx), (F_O_idx, L_X_idx)),
        ((F_N_idx, F_p_NDir_idx), (F_N_idx, L_X_idx)),
        ((L_X_idx, L_C1_idx), (F_O_idx, L_X_idx)),
        ((L_X_idx, L_C1_idx), (F_N_idx, L_X_idx)),
    )

    fixed_coords = np.vstack([F_O_coord, F_C_coord, F_N_coord, F_p_NDir_coord])
    varying_coords = np.hstack([L_X_coords, L_C1_coords]).reshape(-1, 2, 3)
    distance_features, angle_features = get_distances_and_angles(fixed_coords, varying_coords, distance_features, angle_features, None)

    # invert distances
    distance_features = 6 - distance_features
    distance_features = np.clip(distance_features, 0, 6)

    
    # halogen features
    chlorine = 1 if halogen_type == "Cl" else 0; bromine = 1 if halogen_type == "Br" else 0; iodine = 1 if halogen_type == "I" else 0
    halogen_features = np.full((len(varying_coords), 3), [chlorine, bromine, iodine])

    return np.hstack([distance_features, angle_features, halogen_features])

def get_bb_head_on_features_from_molecule(rdkit_mol, translations = None):
    """ Computes features for the bb_head_on_model from the given rdkit molecule. Optionally returns the features for multiple translations.

    Parameters
    ----------
    rdkit_mol : rdkit.Chem.rdchem.Mol
        The molecule to compute features for.
    translations : np.ndarray, optional
        The translations to apply to the molecule. Should be of shape (n_variations, 3). The default is [[0,0,0]].
    

    Returns
    -------
    np.ndarray
        The computed features. Should be of shape (n_variations, 9).
    """
    
    if translations is None:
        translations = np.array([[0,0,0]])

    # determine atoms
    F_O = [atom for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() == "O"][0]
    
    L_X = [atom for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() in ["Cl", "Br", "I"]][0]
    L_C1 = L_X.GetNeighbors()[0]
    L_X_coords = rdutil.position(L_X)
    L_C1_coords = rdutil.position(L_C1)
    L_X_coords = L_X_coords[None, :] + translations
    L_C1_coords = L_C1_coords[None, :] + translations

    return get_bb_head_on_features_from_acceptor_oxygen(F_O, L_X_coords, L_C1_coords, L_X.GetSymbol())

def get_bb_head_on_features_from_acceptor_oxygen(oxygen, halogen_coords, halogen_neighbor_coords, halogen_type):
    """ Computes features for the bb_side_on_model from the given coordinates. And the given acceptor oxygens.
    
    Parameters
    ----------
    oxygen : rdkit.Chem.rdchem.Atom
        The acceptor oxygen.
    halogen_coords : np.ndarray
        Coordinates of the X atom of the ligand. Should be of shape (n_variations, 3).
    halogen_neighbor_coords : np.ndarray
        Coordinates of the C1 atom of the ligand. Should be of shape (n_variations, 3).
    halogen_type : str
        The type of halogen. Should be one of "Cl", "Br" or "I".

    Returns
    -------
    np.ndarray
        The computed features. Should be of shape (n_variations, 9).
    """
    F_O = oxygen
    F_C = F_O.GetNeighbors()[0]
    F_N = [atom for atom in F_C.GetNeighbors() if atom.GetSymbol() == "N"][0]
    F_N_heavy_neighbors = [a for a in F_N.GetNeighbors()]  # [a for a in F_N.GetNeighbors() if a.GetSymbol() != "H"]
    F_p_NDir_coord = np.mean([rdutil.position(a) for a in F_N_heavy_neighbors], axis = 0)

    L_X_coords = halogen_coords
    L_C1_coords = halogen_neighbor_coords

    F_O_coord, F_C_coord, F_N_coord = rdutil.position([F_O, F_C, F_N])
    return get_bb_head_on_model_features(F_O_coord, F_C_coord, F_N_coord, F_p_NDir_coord, L_X_coords, L_C1_coords, halogen_type)


# =============================================================================
# bb_side_on_model
# =============================================================================

def get_bb_side_on_model_features(F_O_coord, F_C_coord, F_N_coord, F_CA_coord, F_HN_coord, F_CN_coord, F_HCA1_coord, F_HCA2_coord, F_HCA3_coord, F_HCN1_coord, F_HCN2_coord, F_HCN3_coord, L_X_coords, L_C1_coords, halogen_type, just_closest = False):
    """ Computes features for the bb_side_on_model from the given coordinates.
    
    Parameters
    ----------
    F_O_coord : np.ndarray
        Coordinates of the O atom of the fixed group. Should be of shape (3,).
    F_C_coord : np.ndarray
        Coordinates of the C atom of the fixed group. Should be of shape (3,).
    F_N_coord : np.ndarray
        Coordinates of the N atom of the fixed group. Should be of shape (3,).
    F_CA_coord : np.ndarray
        Coordinates of the C-alpha atom of the fixed group. Should be of shape (3,).
    F_HN_coord : np.ndarray
        Coordinates of the HN atom of the fixed group. Should be of shape (3,).
    F_CN_coord : np.ndarray
        Coordinates of the n-terminal C atom of the fixed group. Should be of shape (3,).
    F_HCA1_coord : np.ndarray
        Coordinates of one of the hydrogen atoms attached to the C-alpha atom of the fixed group. Should be of shape (3,).
    F_HCA2_coord : np.ndarray
        Coordinates of one of the hydrogen atoms attached to the C-alpha atom of the fixed group. Should be of shape (3,).
    F_HCA3_coord : np.ndarray
        Coordinates of one of the hydrogen atoms attached to the C-alpha atom of the fixed group. Should be of shape (3,).
    F_HCN1_coord : np.ndarray
        Coordinates of one of the hydrogen atoms attached to the n-terminal C atom of the fixed group. Should be of shape (3,).
    F_HCN2_coord : np.ndarray
        Coordinates of one of the hydrogen atoms attached to the n-terminal C atom of the fixed group. Should be of shape (3,).
    F_HCN3_coord : np.ndarray
        Coordinates of one of the hydrogen atoms attached to the n-terminal C atom of the fixed group. Should be of shape (3,).
    L_X_coords : np.ndarray
        Coordinates of the X atom of the ligand. Should be of shape (n_variations, 3).
    L_C1_coords : np.ndarray
        Coordinates of the C1 atom of the ligand. Should be of shape (n_variations, 3).
    halogen_type : str
        The type of halogen. Should be one of "Cl", "Br" or "I".
    just_closest : bool, optional
        Whether to only return the closest distance and angle features for equivalent atoms. The default is False.

    Returns
    -------
    np.ndarray
        The computed features. Should be of shape (n_variations, 19).
   """

    # distance features:
    F_O_idx, F_C_idx, F_N_idx, F_CA_idx, F_HN_idx, F_CN_idx, F_HCA1_idx, F_HCA2_idx, F_HCA3_idx, F_HCN1_idx, F_HCN2_idx, F_HCN3_idx, L_X_idx, L_C1_idx = np.arange(14)
    distance_features = [(fixed, L_X_idx) for fixed in [F_O_idx, F_C_idx, F_N_idx, F_CA_idx, F_HN_idx, F_CN_idx, F_HCA1_idx, F_HCA2_idx, F_HCA3_idx, F_HCN1_idx, F_HCN2_idx, F_HCN3_idx]]
    angle_features = [(fixed_angle, (L_X_idx, L_C1_idx)) for fixed_angle in distance_features]

    fixed_coords = np.vstack([F_O_coord, F_C_coord, F_N_coord, F_CA_coord, F_HN_coord, F_CN_coord, F_HCA1_coord, F_HCA2_coord, F_HCA3_coord, F_HCN1_coord, F_HCN2_coord, F_HCN3_coord])
    varying_coords = np.hstack([L_X_coords, L_C1_coords]).reshape(-1, 2, 3)

    equivalent_atoms = (F_HCA1_idx, F_HCA2_idx, F_HCA3_idx), (F_HCN1_idx, F_HCN2_idx, F_HCN3_idx)

    distance_features, angle_features = get_distances_and_angles(fixed_coords, varying_coords, distance_features, angle_features, equivalent_atoms)

    if just_closest:
        distance_features = np.delete(distance_features, [F_HCA2_idx, F_HCA3_idx, F_HCN2_idx, F_HCN3_idx], axis = 1)
        angle_features = np.delete(angle_features, [F_HCA2_idx, F_HCA3_idx, F_HCN2_idx, F_HCN3_idx], axis = 1)
        # invert distances
    distance_features = 5 - distance_features
    distance_features = np.clip(distance_features, 0, 5)
    
    # halogen features
    chlorine = 1 if halogen_type == "Cl" else 0; bromine = 1 if halogen_type == "Br" else 0; iodine = 1 if halogen_type == "I" else 0
    halogen_features = np.full((len(varying_coords), 3), [chlorine, bromine, iodine])

    return np.hstack([distance_features, angle_features, halogen_features])




def get_bb_side_on_features_from_molecule(rdkit_mol, translations = None, just_closest = False):
    """ Computes features for the bb_side_on_model from the given rdkit molecule. Optionally returns the features for multiple translations.

    Parameters
    ----------
    rdkit_mol : rdkit.Chem.rdchem.Mol
        The molecule to compute features for.
    translations : np.ndarray, optional
        The translations to apply to the molecule. Should be of shape (n_variations, 3). The default is [[0,0,0]].
    

    Returns
    -------
    np.ndarray
        The computed features. Should be of shape (n_variations, 19).
    """
    
    if translations is None:
        translations = np.array([[0,0,0]])

    # determine atoms
    L_X = [atom for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() in ["Cl", "Br", "I"]][0]
    L_X_coords = rdutil.position(L_X)
    L_C1 = L_X.GetNeighbors()[0]
    L_C1_coords = rdutil.position(L_C1)
    F_O = [atom for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() == "O"][0]
    
    # translate all ligand_coordinates
    L_X_coords = L_X_coords[None, :] + translations
    L_C1_coords = L_C1_coords[None, :] + translations

    return get_bb_side_on_features_from_acceptor_oxygen(F_O, L_X_coords, L_C1_coords, L_X.GetSymbol(), just_closest)

def get_bb_side_on_features_from_acceptor_oxygen(oxygen, halogen_coords, halogen_neighbor_coords, halogen_type, from_protein = False, just_closest = False):
    """ Computes features for the bb_side_on_model from the given coordinates. And the given acceptor oxygens.
    
    Parameters
    ----------
    oxygen : rdkit.Chem.rdchem.Atom
        The acceptor oxygen.
    halogen_coords : np.ndarray
        Coordinates of the X atom of the ligand. Should be of shape (n_variations, 3).
    halogen_neighbor_coords : np.ndarray
        Coordinates of the C1 atom of the ligand. Should be of shape (n_variations, 3).
    halogen_type : str
        The type of halogen. Should be one of "Cl", "Br" or "I".
    from_protein : bool, optional
        Whether to ignore warnings about missing hydrogens. The default is False.
    just_closest : bool, optional
        Whether to only return the closest distance and angle features for equivalent atoms. The default is False.
        
    Returns
    -------
    np.ndarray
        The computed features. Should be of shape (n_variations, 19).
    """
    F_O = oxygen
    F_C = F_O.GetNeighbors()[0]
    F_N = [atom for atom in F_C.GetNeighbors() if atom.GetSymbol() == "N"][0]
    F_CA = [atom for atom in F_C.GetNeighbors() if atom.GetSymbol() == "C"][0]
    F_HN = [atom for atom in F_N.GetNeighbors() if atom.GetSymbol() == "H"]
    F_CN = [atom for atom in F_N.GetNeighbors() if (atom.GetSymbol() == "C") and not (atom.GetIdx() == F_C.GetIdx())][0]
    F_N_heavy_neighbors = [a for a in F_N.GetNeighbors()]  # [a for a in F_N.GetNeighbors() if a.GetSymbol() != "H"]
    F_HCA = [atom for atom in F_CA.GetNeighbors() if atom.GetSymbol() == "H"]
    F_HCN = [atom for atom in F_CN.GetNeighbors() if atom.GetSymbol() == "H"]

    L_X_coords = halogen_coords
    L_C1_coords = halogen_neighbor_coords
    fixed_coordinates = np.zeros((3, 12))
    fixed_coordinates[[0, 1, 2, 3, 5]] = rdutil.position([F_O, F_C, F_N, F_CA, F_CN])
    if len(F_HN) > 0:
        fixed_coordinates[4] = rdutil.position(F_HN[0])
    else:
        if not from_protein: logging.warning("No hydrogen found on N atom, inferring position. If you want to ignore this set from_protein = True.")
        fixed_coordinates[4] = _infer_single_hydrogen_position(F_N)
    fixed_coordinates[6:len(F_HCA)] = np.array([rdutil.position(a) for a in F_HCA])
    if (not from_protein) and len(F_HCA) < 3: logging.warning("Less then 3 hydrogens found on CA. Setting missing hydrogens to [0,0,0]. If you want to ignore this set from_protein = True.")
    fixed_coordinates[9:len(F_HCN)] = np.array([rdutil.position(a) for a in F_HCN])
    if (not from_protein) and len(F_HCN) < 3: logging.warning("Less then 3 hydrogens found on CN. Setting missing hydrogens to [0,0,0]. If you want to ignore this set from_protein = True.")

    return get_bb_side_on_model_features(*fixed_coordinates, L_X_coords, L_C1_coords, halogen_type, just_closest)

