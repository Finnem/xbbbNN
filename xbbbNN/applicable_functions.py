from xbpy import rdutil
import numpy as np
def get_xb_bb_positions(oxygen_atom):
    """
    Returns the positions relevant to the heatvolume function for each potential acceptor:
    * the backbone oxygens (O) position
    * the backbone oxygens neighbor (C) position
    * the backbone oxygens C-alpha (CA) position
    * the backbone oxygens H-alpha (HA) position
    * the nitrogens (N) position
    * the nitrogens hydrogen (H) position
    * the nitrogens neighbor (CA) position
    * the nitrogens neighbors hydrogen (HA) position
    """
    # selecting the halogen atom
    
    positions = []
    positions.append(rdutil.position(oxygen_atom))
    oxygen_neighbor = oxygen_atom.GetNeighbors()[0]
    positions.append(rdutil.position(oxygen_neighbor))
    ca_immediate = [a for a in oxygen_neighbor.GetNeighbors() if a.GetSymbol() == "C"][0]
    positions.append(rdutil.position(ca_immediate))

    #HA
    ca_immediate_hydrogen_neighbors = [a for a in ca_immediate.GetNeighbors() if a.GetSymbol() == "H"]
    if len(ca_immediate_hydrogen_neighbors) == 0:
        # inferring HA position
        positions.append(_infer_single_hydrogen_position(ca_immediate))
    elif len(ca_immediate_hydrogen_neighbors) == 1:
        positions.append(rdutil.position(ca_immediate_hydrogen_neighbors[0]))
    else:
        # determine closest hydrogen
        HA1 = rdutil.position(ca_immediate_hydrogen_neighbors[0])
        HA2 = rdutil.position(ca_immediate_hydrogen_neighbors[1])
        if np.linalg.norm(HA1 - positions[0]) < np.linalg.norm(HA2 - positions[0]):
            positions.append(HA1)
        else:
            positions.append(HA2)

    n = [a for a in oxygen_neighbor.GetNeighbors() if a.GetSymbol() == "N"][0]
    positions.append(rdutil.position(n))

    n_hydrogen_neighbors = [a for a in n.GetNeighbors() if a.GetSymbol() == "H"]
    if len(n_hydrogen_neighbors) == 0:
        # inferring HA position
        positions.append(_infer_single_hydrogen_position(n))
    elif len(n_hydrogen_neighbors) == 1:
        positions.append(rdutil.position(n_hydrogen_neighbors[0]))
    else:
        raise NotImplementedError("Nitrogen with more than one hydrogen neighbor")
    
    ca_n = [a for a in n.GetNeighbors() if (a.GetSymbol() == "C") and (not (a.GetIdx() == oxygen_neighbor.GetIdx()))][0]
    positions.append(rdutil.position(ca_n))

    ca_n_hydrogen_neighbors = [a for a in ca_n.GetNeighbors() if a.GetSymbol() == "H"]
    if len(ca_n_hydrogen_neighbors) == 0:
        # inferring HA position
        positions.append(_infer_single_hydrogen_position(ca_n))
    elif len(ca_n_hydrogen_neighbors) == 1:
        positions.append(rdutil.position(ca_n_hydrogen_neighbors[0]))
    else:
        # determine closest hydrogen
        HA1 = rdutil.position(ca_n_hydrogen_neighbors[0])
        HA2 = rdutil.position(ca_n_hydrogen_neighbors[1])
        if np.linalg.norm(HA1 - positions[0]) < np.linalg.norm(HA2 - positions[0]):
            positions.append(HA1)
        else:
            positions.append(HA2)
    return positions


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

def get_network_features(halogen_type, halogen_position, carbon_position, acceptor_positions):
    X_to_O = np.array(acceptor_positions[0]) - np.array(halogen_position) 
    X_to_C_O = np.array(acceptor_positions[1]) - np.array(halogen_position) 
    X_to_N_O = np.array(acceptor_positions[4]) - np.array(halogen_position) 
    X_to_C_C_O = np.array(acceptor_positions[2]) - np.array(halogen_position) 
    X_to_H_N_O = np.array(acceptor_positions[5]) - np.array(halogen_position) 
    X_to_C_N_O = np.array(acceptor_positions[6]) - np.array(halogen_position) 
    X_to_H_C_C_O = np.array(acceptor_positions[3]) - np.array(halogen_position) 
    X_to_H_C_N_O = np.array(acceptor_positions[7]) - np.array(halogen_position) 

    XC_to_X = np.array(halogen_position) - np.array(carbon_position); XC_to_X /= np.linalg.norm(XC_to_X)

    chlorine = 1 if halogen_type == "Cl" else 0; bromine = 1 if halogen_type == "Br" else 0; iodine = 1 if halogen_type == "I" else 0
    CO_to_O = np.array(acceptor_positions[0]) - np.array(acceptor_positions[1]); CO_to_O /= np.linalg.norm(CO_to_O)
    BB_to_N = np.array(acceptor_positions[4]) - np.array([acceptor_positions[2], acceptor_positions[6]]).mean(axis = 0); BB_to_N /= np.linalg.norm(BB_to_N)

    distances = np.array([np.linalg.norm(X_to_O), np.linalg.norm(X_to_C_O), np.linalg.norm(X_to_N_O), np.linalg.norm(X_to_C_C_O), np.linalg.norm(X_to_H_N_O), np.linalg.norm(X_to_C_N_O), np.linalg.norm(X_to_H_C_C_O), np.linalg.norm(X_to_H_C_N_O)])
    distances = np.clip(6 - distances, 0.0, 6.0)

    head_on_input = np.array([distances[0], distances[2], np.dot(X_to_O, CO_to_O), np.dot(X_to_N_O, BB_to_N), np.dot(X_to_O, XC_to_X), np.dot(X_to_N_O, XC_to_X), chlorine, bromine, iodine])

    side_on_input = np.array([*np.clip(distances - 1, 0, 5), np.dot(X_to_O, XC_to_X), np.dot(X_to_C_O, XC_to_X), np.dot(X_to_N_O, XC_to_X), np.dot(X_to_C_C_O, XC_to_X), np.dot(X_to_H_N_O, XC_to_X), np.dot(X_to_C_N_O, XC_to_X), np.dot(X_to_H_C_C_O, XC_to_X), np.dot(X_to_H_C_N_O, XC_to_X), chlorine, bromine, iodine])

    return head_on_input, side_on_input

def apply_small_network_explicit(features):
    SmallSideOnInputGate = np.array([0.053437013, 0.08720989, 0.03499039, 0.17911866, 0.052967586, 0.14525391, 0.3594176, 0.15516634, -0.00030686276, 0.090750456, 0.000516477, 0.1443829, 0.051579367, 0.055814285, 0.01121225, 0.030701246, 0.013733487, 0.00082908425, 0.043782104])
    SmallSideOnDenseKernel0 = np.array([-1.0945588e+00, 1.0210054e+00, -8.9513443e-02, 4.9032667e-01,
        3.7412286e-01, 8.8899440e-01, 7.5313997e-01, 5.8522290e-01,
        1.6661527e-02, 1.4660324e+00, 1.0658792e-01, 2.5994641e-01,
        8.6569899e-01, 6.2613070e-02, 7.5808454e-01, 3.5833764e-01,
        -5.6344199e-01, -3.5331437e-01, -2.8002593e-01, -1.0302281e-01,
        -1.0439985e+00, -9.4499379e-02, -5.9990603e-01, 4.7993109e-01,
        -6.5267771e-01, -5.5089027e-01, 2.4496805e-02, 7.8488594e-01,
        7.1581775e-01, -1.8345164e-01, 1.0331861e+00, 5.5494511e-01,
        -5.4455437e-03, 2.6009655e-02, 2.2769960e-02, 1.9436706e-02,
        4.5064425e-01, 4.0482575e-01, -5.4065520e-01, 1.0067091e+00,
        -3.9523421e-04, 3.8627163e-02, 2.0865833e-02, 1.5116045e-02,
        -1.3039041e-01, -3.4704685e-01, 4.9508825e-01, -6.9578272e-01,
        7.7305876e-02, 4.9246839e-01, -5.7117409e-01, 2.3703781e-01,
        -6.2906772e-02, -6.5052336e-01, 7.9904157e-01, -6.8191177e-01,
        7.8095056e-02, 1.2951745e+00, 9.2151624e-01, -9.5859706e-01,
        -4.4021145e-01, 5.2530318e-01, 5.1792520e-01, -1.5709822e+00,
        6.1700064e-01, -6.2226713e-01, -5.4225016e-01, -1.5708187e-01,
        8.1839688e-02, 5.7985508e-01, 4.2778078e-01, 2.0169061e-01,
        -2.3941879e-01, 5.4919201e-01, 8.9173353e-01, 6.2243617e-01]).reshape((19, 4))
    SmallSideOnDenseBias0 = np.array([-0.02385203, 0.00485941, -0.2992362, -0.50467056]).reshape((4,))
    SmallSideOnDenseKernel1 = np.array([ -4.234208, -0.578204, 1.1886917, -0.84875584]).reshape((4, 1))
    SmallSideOnDenseBias1 = np.array([-0.06216126]).reshape((1,))
    SmallSideOnDenseKernel2 = np.array([5.681173]).reshape((1, 1))
    SmallSideOnDenseBias2 = np.array([-0.09458347]).reshape((1,))

    SmallHeadOnInputGate = np.array([0.18970639, 0.0244453, 0.05529752, 0.04794552, 0.23057207,
	0.05515747, 0.03252049, 0.00102495, 0.07287373])
    SmallHeadOnDenseKernel0 = np.array([0.7895073, -1.8237941, 0.846557, 0.9866653, -0.586774,
	1.691605, -1.2332207, 0.4557941, -1.1539291, -0.4966678,
	0.40302858, -1.7995813, 0.06012528, 2.9843743, 0.02247606,
	-0.09397221, -0.04202208, -1.59293]).reshape((9, 2))
    SmallHeadOnDenseBias0 = np.array([-0.19576368, 0.01510766]).reshape((2,))
    SmallHeadOnDenseKernel1 = np.array([-0.8201565, -1.043865, -1.3877711, -1.1556886]).reshape((2, 2))
    SmallHeadOnDenseBias1 = np.array([0.20289162, -0.23238967]).reshape((2,))
    SmallHeadOnDenseKernel2 =np.array([1.6438103, 0.9480839, -2.2671022, 4.0970893]).reshape((2, 2))
    SmallHeadOnDenseBias2 = np.array([-0.76008344, -0.35954875]).reshape((2,))
    SmallHeadOnDenseKernel3 = np.array([5.395625, -2.962778]).reshape((2, 1))
    SmallHeadOnDenseBias3 = np.array([0.02771468])

    head_on_input = features[0]
    side_on_input = features[1]

    head_on_kernels = [SmallHeadOnDenseKernel0, SmallHeadOnDenseKernel1, SmallHeadOnDenseKernel2, SmallHeadOnDenseKernel3]
    head_on_biases = [SmallHeadOnDenseBias0, SmallHeadOnDenseBias1, SmallHeadOnDenseBias2, SmallHeadOnDenseBias3]
    head_on_input_gate = SmallHeadOnInputGate

    side_on_kernels = [SmallSideOnDenseKernel0, SmallSideOnDenseKernel1, SmallSideOnDenseKernel2]
    side_on_biases = [SmallSideOnDenseBias0, SmallSideOnDenseBias1, SmallSideOnDenseBias2]
    side_on_input_gate = SmallSideOnInputGate

    head_on_result = head_on_input_gate * head_on_input
    for kernel, bias in zip(head_on_kernels, head_on_biases):
        head_on_result = np.matmul(head_on_result.T, kernel) + bias.T
        head_on_result[head_on_result < 0] = head_on_result[head_on_result < 0] * 0.2

    side_on_result = side_on_input_gate * side_on_input
    for kernel, bias in zip(side_on_kernels, side_on_biases):
        side_on_result = np.matmul(side_on_result.T, kernel) + bias.T
        side_on_result[side_on_result < 0] = side_on_result[side_on_result < 0] * 0.2

    return head_on_result, side_on_result

def small_network_evaluation(halogen_position, orientation, acceptor_atoms, halogen_symbol, bond_distance, explicit = True):
    """
    Evaluates the small network on a single example
    """
    summed_head_on_result = np.array([0.0])
    summed_side_on_result = np.array([0.0])
    for acceptor_atom in acceptor_atoms:
        try:
            acceptor_positions = get_xb_bb_positions(acceptor_atom)
        except IndexError:
            continue
        carbon_position = halogen_position - (orientation * bond_distance)
        head_on_input, side_on_input = get_network_features(halogen_symbol, halogen_position, carbon_position, acceptor_positions)

        if explicit:
            head_on_result, side_on_result = apply_small_network_explicit([head_on_input, side_on_input])
        else:
            head_on_result, side_on_result = apply_small_network([head_on_input, side_on_input])
        summed_head_on_result += head_on_result
        summed_side_on_result += side_on_result
    return summed_head_on_result + summed_side_on_result

def head_on_network_evaluation(halogen_position, orientation, acceptor_atoms, halogen_symbol, bond_distance, explicit = True):
    """
    Evaluates the small network on a single example
    """
    summed_head_on_result = [0.0]
    summed_side_on_result = [0.0]
    for acceptor_atom in acceptor_atoms:
        try:
            acceptor_positions = get_xb_bb_positions(acceptor_atom)
        except IndexError:
            continue
        carbon_position = halogen_position - (orientation * bond_distance)
        head_on_input, side_on_input = get_network_features(halogen_symbol, halogen_position, carbon_position, acceptor_positions)
        
        if explicit:
            head_on_result, side_on_result = apply_small_network_explicit([head_on_input, side_on_input])
        else:
            head_on_result, side_on_result = apply_small_network([head_on_input, side_on_input])
        summed_head_on_result.append(head_on_result[0])
        summed_side_on_result.append(side_on_result[0])
    return np.max(summed_head_on_result)