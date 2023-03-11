import os
from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors
from metrics.SA_Score import sascorer
from metrics.NP_Score import npscorer
from utils import mapper

_base_dir = os.path.split(__file__)[0]
_mcf = pd.read_csv(os.path.join(_base_dir, 'mcf.csv'))
_pains = pd.read_csv(os.path.join(_base_dir, 'pains.csv'),
                     names=['smarts', 'names'])
_filters = [Chem.MolFromSmarts(x) for x in pd.concat([_mcf, _pains], sort=True)['smarts'].values]


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def standardize_smiles(smiles, basicClean=True, clearCharge=True, clearFrag=True, canonTautomer=False, isomeric=True):
    try:
        clean_mol = Chem.MolFromSmiles(smiles)
        if basicClean:
        # 除去氢、金属原子、标准化分子
            clean_mol = rdMolStandardize.Cleanup(clean_mol)
        if clearFrag:
        # 仅保留主要片段作为分子
            clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        if clearCharge:
        # 尝试中性化处理分子
            uncharger = rdMolStandardize.Uncharger() 
            clean_mol = uncharger.uncharge(clean_mol)
        if canonTautomer:
        # 处理互变异构情形，得到的结构不一定是最稳定的构型
            te = rdMolStandardize.TautomerEnumerator() # idem
            clean_mol = te.Canonicalize(clean_mol)
        # 是否保留立体信息，并将分子存为标准化后的SMILES形式
        std_smiles=Chem.MolToSmiles(clean_mol, isomericSmiles=isomeric)
    except Exception as e:
        print (e, smiles)
        return None
    return std_smiles


def logP(mol):
    """
    Computes RDKit's logP
    """
    return Chem.Crippen.MolLogP(mol)


def SA(mol):
    """
    Computes RDKit's Synthetic Accessibility score
    """
    return sascorer.calculateScore(mol)


def NP(mol):
    """
    Computes RDKit's Natural Product-likeness score
    """
    return npscorer.scoreMol(mol)


def QED(mol):
    """
    Computes RDKit's QED score
    """
    return qed(mol)


def weight(mol):
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    return Descriptors.MolWt(mol)


def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()


def fragmenter(mol):
    """
    fragment mol using BRICS and return smiles list
    """
    fgs = AllChem.FragmentOnBRICSBonds(get_mol(mol))
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi

def compute_fragments(mol_list, n_jobs=1):
    """
    fragment list of mols using BRICS and return smiles list
    """
    fragments = []
    for mol_frag in mapper(n_jobs)(fragmenter, mol_list):
        fragments.extend(mol_frag)
    return list(set(fragments))


def compute_scaffold(mol, min_rings=2):
    """
    Extracts a scafold from a molecule in a form of a canonic SMILES
    """
    mol = get_mol(mol)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles

def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    scaffolds = []
    for mol_scaf in mapper(n_jobs)(compute_scaffold, mol_list):
        if mol_scaf is not None:
            scaffolds.append(mol_scaf)
    return list(set(scaffolds))

def calc_self_tanimoto(gen_vecs, agg='max', device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules
    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    
    # Initialize output array and total count for mean aggregation
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    
    # Convert input vectors to PyTorch tensors and move to the specified device
    x_gen = torch.tensor(gen_vecs).to(device).half()
    y_gen = torch.tensor(gen_vecs).to(device).half()
            
    # Transpose x_stock tensor
    y_gen = y_gen.transpose(0, 1)
    
    # Calculate Tanimoto similarity using matrix multiplication
    tp = torch.mm(x_gen, y_gen)
    jac = (tp / (x_gen.sum(1, keepdim=True) + y_gen.sum(0, keepdim=True) - tp))
            
    # Handle NaN values in the Tanimoto similarity matrix
    jac = jac.masked_fill(torch.isnan(jac), 1)
    
    # Delete the elements on the eye (self-self similarity)
    jac = jac[~np.eye(jac.shape[0], dtype=bool)].reshape(jac.shape[0],-1)
            
    if p != 1:
        jac = jac**p
            
    # Aggregate scores from this batch
    if agg == 'max':
        # Aggregate using max
        agg_tanimoto = jac.max(1)[0].cpu().numpy()
    elif agg == 'mean':
        # Aggregate using mean
        agg_tanimoto = jac.mean(1).cpu().numpy()
        
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    
    return agg_tanimoto


def calc_agg_tanimoto(gen_vecs, stock_vecs, batch_size=5000, agg='max', device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules
    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    
    # Initialize output array and total count for mean aggregation
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    
    # Convert input vectors to PyTorch tensors and move to the specified device
    x_stock = torch.tensor(stock_vecs).to(device).half()
    y_gen = torch.tensor(gen_vecs).to(device).half()
    
    # Loop over batches of stock vectors
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_batch = x_stock[j:j + batch_size]
        
        # Loop over batches of generated vectors
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_batch = y_gen[i:i + batch_size]
            
            # Transpose x_stock tensor
            y_batch = y_batch.transpose(0, 1)
    
            # Calculate Tanimoto similarity using matrix multiplication
            tp = torch.mm(x_batch, y_batch)
            jac = (tp / (x_batch.sum(1, keepdim=True) + y_batch.sum(0, keepdim=True) - tp))
            
            # Handle NaN values in the Tanimoto similarity matrix
            jac = jac.masked_fill(torch.isnan(jac), 1)
            
            if p != 1:
                jac = jac**p
            
            # Aggregate scores from this batch
            if agg == 'max':
                # Aggregate using max
                agg_tanimoto[i:i + y_batch.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_batch.shape[1]], jac.max(0)[0].cpu().numpy())
            elif agg == 'mean':
                # Aggregate using mean
                agg_tanimoto[i:i + y_batch.shape[1]] += jac.mean(0).cpu().numpy()
                total[i:i + y_batch.shape[1]] += jac.shape[0]
    
    # Compute average score for mean aggregation
    if agg == 'mean':
        agg_tanimoto /= total
    
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return agg_tanimoto


def fingerprint(smiles_or_mol, fp_type='morgan', dtype=None, morgan__r=2, morgan__n=1024):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits
    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n),
                                 dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint


def fingerprints(smiles_mols_array, n_jobs=1, already_unique=True, **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    '''
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array, return_inverse=True)

    fps = mapper(n_jobs)(fingerprint, smiles_mols_array)

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
           for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps


def mol_passes_filters(mol,
                       allowed=None,
                       isomericSmiles=False):
    """
    Checks if mol
    * passes MCF and PAINS filters,
    * has only allowed atoms
    * is not charged
    """
    allowed = allowed or {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}
    mol = get_mol(mol)
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(
            len(x) >= 8 for x in ring_info.AtomRings()
    ):
        return False
    h_mol = Chem.AddHs(mol)
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    if smiles is None or len(smiles) == 0:
        return False
    if Chem.MolFromSmiles(smiles) is None:
        return False
    return True
