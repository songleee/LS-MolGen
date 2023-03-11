import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from multiprocessing import Pool
import os
import re
import shutil
import linecache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, ALL_COMPLETED


def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return (tensor).cuda()
    return torch.autograd.Variable(tensor)

def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)

def seq_to_smiles(seqs, voc):
    """Takes an output sequence from the RNN and returns the
       corresponding SMILES."""
    smiles = []
    for seq in seqs.cpu().numpy():
        smiles.append(voc.decode(seq))
    return smiles

def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    for smile in smiles:
        if valid_smiles(smile):
            i += 1
    return i / len(smiles)

def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))

def valid_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:          # check validity
        return False
    try:                     # check valence, aromaticity, conjugation and hybridization
        Chem.SanitizeMol(mol)
    except:
        return False
    return True

def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


def prepare_receptor(pdb, dock_file_dir, rmsd=1.0, binding_pose_num=20):
    """prepare receptor by lepro"""
    if dock_file_dir is None:
        print('Please select \'dock_file_dir\' for saving output dock files !')
        return
    root_dir = os.path.abspath('.') # get the present dir path
    if not os.path.exists(dock_file_dir):
        os.makedirs(dock_file_dir)
    pdb_name = os.path.basename(pdb)
    if not os.path.isfile(os.path.join(dock_file_dir, pdb_name)):
        shutil.copy(pdb, dock_file_dir)
    os.chdir(dock_file_dir)
    os.system('lepro {}'.format(pdb_name))
    os.system('sed -i \'5s/.*/{}/\' dock.in'.format(rmsd))
    os.system('sed -i \'13s/.*/{}/\' dock.in'.format(binding_pose_num))
    os.chdir(root_dir)
    print('Docking files (pro.pdb, dock.in) were generated to {}'.format(dock_file_dir))
    return

def get_ledock_score(smiles, dock_file_dir='./data/ledock', work_dir='./data/ledock_1', save_work_dir=True):
    """Docking scores based on Ledock"""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    assert dock_file_dir != work_dir
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    try:
        shutil.copy(os.path.join(dock_file_dir, "pro.pdb"), work_dir)
        shutil.copy(os.path.join(dock_file_dir, "dock.in"), work_dir)
    except Exception as r:
        print(r)
    os.chdir(work_dir)
    if isinstance(smiles, str):
        try:
            os.system('obabel -:\'{}\' -omol2 -O ./lig.mol2 --gen3D > /dev/null 2>&1'.format(smiles))
            write_smiles_to_file(['./lig.mol2'], './ligands')
            os.system('ledock ./dock.in')
            line_docking_score = linecache.getline('./lig.dok', 2)
            rex = re.search('Score: (.+) kcal/mol', line_docking_score)
            docking_score = float(rex.group(1))  # match the score in the first ()
            os.chdir(root_dir)
            return docking_score
        except Exception as r:
            print(r)
            os.chdir(root_dir)
            return 0.0

    _smiles = [smi for i, smi in enumerate(smiles) if valid_smiles(smi)]
    _id = [i for i, smi in enumerate(smiles) if valid_smiles(smi)]
    write_smiles_to_file(_smiles, './lig.smi')
    os.system('obabel ./lig.smi -omol2 -O ./lig.mol2 --gen3D -m > /dev/null 2>&1')
    # mol_list = [filename for filename in os.listdir('.') if filename.endswith('mol2')]
    mol_list = ['./lig{}.mol2'.format(i + 1) for i in range(len(_smiles))]
    write_smiles_to_file(mol_list, './ligands')
    os.system('ledock ./dock.in')
    score = []
    j = 0
    for i in range(len(smiles)):
        if i not in _id:
            score.append(0.0)
            continue
        line_docking_score = linecache.getline('./lig{}.dok'.format(j+1), 2)
        j = j + 1
        rex = re.search('Score: (.+) kcal/mol', line_docking_score)
        if not rex:
            score.append(0.0)
        else:
            score.append(float(rex.group(1)))  # match the score in the first ()
    os.chdir(root_dir)
    if not save_work_dir:
        os.system("rm -r {}".format(work_dir))
    return score


def get_ledock_score_parallel(smiles:list, n=32, pool='process', 
                    dock_file_dir='./data/ledock', work_dir='./data/ledock_', 
                    save_work_dir=False):
    scores = []
    smiles_list = np.array_split(smiles, n)
    if pool=='process':
        with ProcessPoolExecutor(max_workers=n) as executor:
            tasks = [executor.submit(get_ledock_score, Smiles, dock_file_dir, work_dir+str(j), save_work_dir) \
                  for j, Smiles in enumerate(smiles_list)]
            wait(tasks, return_when=ALL_COMPLETED)
            for res in tasks:
                scores.extend(res.result())
    elif pool=='thread':
        with ThreadPoolExecutor(max_workers=n) as executor:
            tasks = [executor.submit(get_ledock_score, Smiles, dock_file_dir, work_dir+str(j), save_work_dir) \
                  for j, Smiles in enumerate(smiles_list)]
            wait(tasks, return_when=ALL_COMPLETED)
            for res in tasks:
                scores.extend(res.result())
    else:
        print('please choose the pool type between \'process\' and \'thread\'.')
        return None
    #print(scores)
    return scores


def disable_rdkit_log():
    rdBase.DisableLog('rdApp.*')

def enable_rdkit_log():
    rdBase.EnableLog('rdApp.*')

def read_smiles_csv(path, sep=','):
    return pd.read_csv(path, usecols=['Smiles'], sep=sep).squeeze('columns').astype(str).tolist()

def read_score_csv(path, sep=','):
    return pd.read_csv(path, usecols=['Ledock'], sep=sep).squeeze('columns').astype(float).tolist()

def write_smiles_to_file(smiles_list, fname):
    """Write a list of SMILES to a file."""
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")

