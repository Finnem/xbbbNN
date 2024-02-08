import numpy as np
from xbpy import rdutil


def _infer_single_hydrogen_position(atom, length = None):
    """
    Infers the hydrogen position of an atom based on the position of its neighbors
    """
    if atom.GetSymbol() == "N":
        length = 1.0
    else:
        length = 1.1
    # inferring HA position
    neighbor_positions = np.array([rdutil.position(a) for a in atom.GetNeighbors()])
    neighbor_positions -= rdutil.position(atom)
    mean_position = neighbor_positions.mean(axis = 0)
    mean_position /= np.linalg.norm(mean_position)
    mean_position *= -length
    inferred_position = rdutil.position(atom) + mean_position
    return inferred_position