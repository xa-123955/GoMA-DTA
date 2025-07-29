import h5py
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from rdkit import Chem
import dgl
from scipy import sparse as sp
import logging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}
# CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
#                  "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
#                  "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
#                  "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
#                  "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
#                  "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
#                  "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
#                  "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def laplacian_positional_encoding(g, pos_enc_dim):
    A = g.adj_external(scipy_fmt='csr')
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g

def atom_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(17),degree(7),formal charge(1),
    radical electrons(1),hybridization(6),aromatic(1),hydrogen atoms attached(5),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']  # 17-dim
    degree = [0, 1, 2, 3, 4, 5, 6,7, 8, 9, 10]  # 11-dim
    implicit_valence = [0, 1, 2, 3, 4, 5, 6]
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']  # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), implicit_valence) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [
                  atom.GetIsAromatic()]  # 17+11+7+1+1+6+1 = 44
    
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4, 5, 6])  # 33+5=38
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 38+3 =41

        # IsHDonor: 判断是否是氢键供体（如 -OH, -NH）
    is_donor = False
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() == 1:  # 氢原子
            is_donor = True
            break
        
        # IsHAcceptor: 判断是否是氢键受体（如 O, N 等）
    is_acceptor = atom.GetAtomicNum() in [7, 8]  # N 或 O
        
    results += [
        is_donor,              # IsHDonor (1)
        is_acceptor,           # IsHAcceptor (1)
        atom.IsInRing(),       # IsInRing (1)
        atom.HasProp('_ChiralityPossible'),  # IsChiral (1)
    ]  # +4 = 58

    # --- 4. Continuous features (normalized to [0,1]) ---
    # Atomic mass (normalized to [0,200])
    mass = atom.GetMass() / 200.0
    
    # Electronegativity (Pauling scale, normalized to [0.5,4.0])
    en = {
        'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66
    }.get(atom.GetSymbol(), 2.0)  # Default for other elements
    en_norm = (en - 0.5) / (4.0 - 0.5)
    
    # Covalent radius (normalized to [0.3,2.0])
    covalent_radii = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
        'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39
    }.get(atom.GetSymbol(), 1.0)  # Default
    cov_rad_norm = (covalent_radii - 0.3) / (2.0 - 0.3)
    
    # Van der Waals radius (normalized to [1.0,3.0])
    vdw_radii = {
        'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
        'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
    }.get(atom.GetSymbol(), 1.5)  # Default
    vdw_rad_norm = (vdw_radii - 1.0) / (3.0 - 1.0)
    
    # Partial charge (normalized to [-1.0,1.0])
    partial_charge = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
    pc_norm = (partial_charge + 1.0) / 2.0  # Scale to [0,1]
    
    results += [mass, en_norm, cov_rad_norm, vdw_rad_norm, pc_norm]  # +5 = 63

    return results

def bond_features(bond, use_chirality=True):
    """Generate bond features including bond type(4), conjugated(1), in ring(1), stereo(4)"""
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)

def smiles_to_graph(smiles, explicit_H=False, use_chirality=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    g = dgl.DGLGraph()
    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    atom_feats = np.array([atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
    if use_chirality:
        chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                                  useLegacyImplementation=False)
        chiral_arr = np.zeros([num_atoms, 3])
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1
        atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)

    g.ndata["atom"] = torch.tensor(atom_feats, dtype=torch.float32)

    # Add edges
    src_list = []
    dst_list = []
    bond_feats_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_feats = bond_features(bond, use_chirality=use_chirality)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)

    g.add_edges(src_list, dst_list)

    g.edata["bond"] = torch.tensor(np.array(bond_feats_all), dtype=torch.float32)
    g = laplacian_positional_encoding(g, pos_enc_dim=8)
    return g

def integer_label_protein(sequence, max_length=998):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in /"
                f"sequence category encoding, skip and treat as " f"padding."
            )
    return encoding


class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.protein_feature_path = "/home/xiong123/L_tt/datasets/2016/features_ESM2_1024_150_mask.h5" 
        self.protein_GO_feature_path = "/home/xiong123/L_tt/datasets/2016/protein_go_description_fixed.h5" 
        self.smiles_to_graph_func = smiles_to_graph
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)


    def _load_protein_features(self):
        """将HDF5文件中的蛋白质特征加载到内存中的字典"""
        protein_features = {}
        with h5py.File(self.protein_feature_path, 'r') as f:
            for protein_id in f['proteins']:
                protein_features[protein_id] = f['proteins'][protein_id][:]
        return protein_features
    
    def _load_protein_GO_features(self):
        """将HDF5文件中的蛋白质特征加载到内存中的字典"""
        protein_features = {}
        with h5py.File(self.protein_GO_feature_path, 'r') as f:
            for protein_id in f['proteins']:  
                embedding = f['proteins'][protein_id][:]  # 读取数据
                # print(f"蛋白质 {protein_id} 读取后的形状: {embedding.shape}")  # 检查形状
                protein_features[protein_id] = embedding
        return protein_features

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        drug = self.df.iloc[index]['Ligand']
        # drug_stand = self.df.iloc[index]['standardized_smiles']
        protein_id = self.df.iloc[index]['uniprot_id']
        with h5py.File(self.protein_feature_path, 'r') as f:
            # v_p = f['proteins'][protein_id][:]
            v_p = f['proteins'][protein_id]['feature'][:]
            v_p_mask = f['proteins'][protein_id]['mask'][:]        
        with h5py.File(self.protein_GO_feature_path, 'r') as f:
            v_2_p = f['proteins'][protein_id][:]
        # Retrieve graph representation for the drug
        v_d_e = self.smiles_to_graph_func(drug)
        # Retrieve the regression label
        y = self.df.iloc[index]["regression_label"]
        return drug, v_d_e, v_p,v_2_p,v_p_mask, y
        # return drug, v_d_e, v_p[0],v_2_p, y