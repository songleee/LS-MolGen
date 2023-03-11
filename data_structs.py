import numpy as np
import pandas as pd
from typing import Tuple, List
import re
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
import torch
from torch.utils.data import Dataset
from utils import Variable

class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""
    
    def __init__(self, init_from_file=None, max_length=140):
        self.special_tokens = ['EOS', 'GO']
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file: self.init_from_file(init_from_file)

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['EOS']: break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = []
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append('EOS')
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters(chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)

class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

        Args:
                fname : path to a file containing \n separated SMILES.
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the Prior.
    """
    def __init__(self, smiles_list, voc):
        self.voc = voc
        self.smiles = smiles_list

    def __getitem__(self, i):
        mol = self.smiles[i]
        tokenized = self.voc.tokenize(mol)
        # enconded
        encoded = self.voc.encode(tokenized)
        return Variable(encoded)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_length = max([seq.size(0) for seq in arr])
        collated_arr = Variable(torch.zeros(len(arr), max_length))
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr

class Inception(object):
    def __init__(self, voc, memory_max_size=5000, min_similarity=0.4, bulket_max_size=1):
        self.memory: pd.DataFrame = pd.DataFrame(columns=['Smiles', 'score', 'likelihood', 'Scaffold', 'Scaf_fp'])
        self.voc = voc
        self.memory_max_size = memory_max_size
        self.min_similarity = min_similarity
        self.bulket_max_size = bulket_max_size

    def add_experience(self, smiles, score, likelihood, scaffold, scaf_fp):
        df = pd.DataFrame({"Smiles": smiles, "score": score, "likelihood": likelihood.detach().cpu().numpy(),\
                     "Scaffold": scaffold, "Scaf_fp": scaf_fp})
        self.memory = self.memory.append(df)
        self._purge_memory()

    def _purge_memory(self):
        unique_df = self.memory.drop_duplicates(subset=["Smiles"])
        sorted_df = unique_df.sort_values('score', ascending=False)
        self.memory = sorted_df.head(self.memory_max_size)
    
    def _load_to_memory(self, scoring_function, prior, smiles):
        if len(smiles):
            self.evaluate_and_add(smiles, scoring_function, prior)

    def evaluate_and_add(self, smiles, scoring_function, prior):
        if len(smiles) > 0:
            score = scoring_function(smiles)
            tokenized = [self.voc.tokenize(smile) for smile in smiles]
            encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
            encoded = MolData.collate_fn(encoded)
            likelihood, _ = prior.likelihood(encoded.long())
            likelihood = likelihood.data.cpu().numpy()
            scaffold = [self._calculate_scaffold(smi) for smi in smiles]
            scaf_fp = [None if scaf is None else Morgan(Chem.MolFromSmiles(scaf), 2, 1024) for scaf in scaffold]
            df = pd.DataFrame({"Smiles": smiles, "score": score, "likelihood": likelihood,\
                         "Scaffold": scaffold, "Scaf_fp": scaf_fp})
            self.memory = self.memory.append(df)
            self._purge_memory()

    def sample(self, sample_size) -> Tuple[List[str], np.array, np.array]:
        sample_size = min(len(self.memory), sample_size)
        if sample_size > 0:
            sampled = self.memory.sample(sample_size)
            smiles = sampled["Smiles"].to_list()
            scores = sampled["score"].to_list()
            prior_likelihood = sampled["likelihood"].to_list()
            tokenized = [self.voc.tokenize(smile) for smile in smiles]
            encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
            encoded = MolData.collate_fn(encoded)
            return encoded, np.array(scores), np.array(prior_likelihood)
        return [], [], []
    
    def save_memory(self, path):
        self.memory[['Smiles', 'score']].to_csv(path, index=None)
        
    def update_score(self, smiles, score):
        scaffold = [self._calculate_scaffold(smi) for smi in smiles]
        scaf_fp = [None if scaf is None else Morgan(Chem.MolFromSmiles(scaf), 2, 1024) for scaf in scaffold]
        similar_scaf = [self._find_similar_scaffold(scaf) for scaf in scaffold]
        scaffold_count = self.memory.Scaffold.dropna().value_counts()
        bulket_size = [scaffold_count[scaf] if scaf in scaffold_count.keys() else 0 for scaf in similar_scaf]
        score = [0 if _size>self.bulket_max_size else _score for _score, _size in zip(score, bulket_size)]  # penalize score
        return np.array(score, dtype=np.float32), scaffold, scaf_fp
        
    def _calculate_scaffold(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
            except ValueError:
                return None
        else:
            return None
        return scaffold_smiles
    
    def _find_similar_scaffold(self, scaffold):
        """
        this function tries to find a "similar" scaffold (according to the threshold set by parameter "minsimilarity") and if at least one
        scaffold satisfies this criteria, it will replace the smiles' scaffold with the most similar one
        -> in effect, this reduces the number of scaffold buckets in the memory (the lower parameter "minsimilarity", the more
           pronounced the reduction)
        generate a "mol" scaffold from the smile and calculate an morgan fingerprint

        :param scaffold: scaffold represented by a smiles string
        :return: closest scaffold given a certain similarity threshold
        """
        if scaffold is not None:
            fp = Morgan(Chem.MolFromSmiles(scaffold), 2, 1024)

            # make a list of the stored fingerprints for similarity calculations
            fps = self.memory.Scaf_fp.dropna().to_list()

            # check, if a similar scaffold entry already exists and if so, use this one instead
            if len(fps) > 0:
                similarity_scores = DataStructs.BulkTanimotoSimilarity(fp, fps)
                closest = np.argmax(similarity_scores)
                if similarity_scores[closest] >= self.min_similarity:
                    scaffold = self.memory.Scaffold.dropna().iloc[closest]
        return scaffold
    
    def __len__(self):
        return len(self.memory)


class Experience(object):
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores."""
    def __init__(self, voc, max_size=5000):
        self.memory = []
        self.max_size = max_size
        self.voc = voc

    def add_experience(self, experience):
        """Experience should be a list of (smiles, score, prior likelihood) tuples"""
        self.memory.extend(experience)
        if len(self.memory)>self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in smiles:
                    idxs.append(i)
                    smiles.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
        # Retain highest scores
        self.memory.sort(key = lambda x: x[1], reverse=True)
        self.memory = self.memory[:self.max_size]
        print("\nBest score in memory: {:.2f}".format(self.memory[0][1]))

    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory)<n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[1] for x in self.memory]
            sample = np.random.choice(len(self.memory), size=n, replace=False, p=scores/np.sum(scores))
            sample = [self.memory[i] for i in sample]
            smiles = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        return encoded, np.array(scores), np.array(prior_likelihood)

    def initiate_from_file(self, fname, scoring_function, Prior):
        """Adds experience from a file with SMILES
           Needs a scoring function and an RNN to score the sequences.
           Using this feature means that the learning can be very biased
           and is typically advised against."""
        with open(fname, 'r') as f:
            smiles = []
            for line in f:
                smile = line.split()[0]
                if Chem.MolFromSmiles(smile):
                    smiles.append(smile)
        scores = scoring_function(smiles)
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        prior_likelihood, _ = Prior.likelihood(encoded.long())
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, scores, prior_likelihood)
        self.add_experience(new_experience)

    def print_memory(self, path):
        """Prints the memory."""
        print("\n" + "*" * 80 + "\n")
        print("         Best recorded SMILES: \n")
        print("Score     Prior log P     SMILES\n")
        with open(path, 'w') as f:
            f.write("Smiles,Score,PriorLogP\n")
            for i, exp in enumerate(self.memory):
                if i < 50:
                    print("{:4.2f}   {:6.2f}        {}".format(exp[1], exp[2], exp[0]))
                f.write("{},{:4.2f},{:6.2f}\n".format(*exp))
        print("\n" + "*" * 80 + "\n")

    def __len__(self):
        return len(self.memory)

def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string

def tokenize(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[xx]' as one token."""
    regex = '(\[[^\[\]]{1,6}\])'
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokenized = []
    for char in char_list:
        if char.startswith('['):
            tokenized.append(char)
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    tokenized.append('EOS')
    return tokenized

def canonicalize_smiles_from_file(fname):
    """Reads a SMILES file and returns a list of RDKIT SMILES"""
    with open(fname, 'r') as f:
        smiles_list = []
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("{} lines processed.".format(i))
            smiles = line.split(" ")[0]
            mol = Chem.MolFromSmiles(smiles)
            if filter_mol(mol):
                smiles_list.append(Chem.MolToSmiles(mol))
        print("{} SMILES retrieved".format(len(smiles_list)))
        return smiles_list

def filter_mol(mol, max_heavy_atoms=50, min_heavy_atoms=10, element_list=[6,7,8,9,15,16,17,35]):
    """Filters molecules on number of heavy atoms and atom types"""
    if mol is not None:
        num_heavy = min_heavy_atoms<mol.GetNumHeavyAtoms()<max_heavy_atoms
        elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        if num_heavy and elements:
            return True
        else:
            return False

def write_smiles_to_file(smiles_list, fname):
    """Write a list of SMILES to a file."""
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")

def filter_on_chars(smiles_list, chars):
    """Filters SMILES on the characters they contain.
       Used to remove SMILES containing very rare/undesirable
       characters."""
    smiles_list_valid = []
    for smiles in smiles_list:
        tokenized = tokenize(smiles)
        if all([char in chars for char in tokenized][:-1]):
            smiles_list_valid.append(smiles)
    print('Filtered library size: %d'%len(smiles_list_valid))
    return smiles_list_valid

def filter_file_on_chars(smiles_fname, voc_fname, fname):
    """Filters a SMILES file using a vocabulary file.
       Only SMILES containing nothing but the characters
       in the vocabulary will be retained."""
    smiles = []
    with open(smiles_fname, 'r') as f:
        for line in f:
            smiles.append(line.split()[0])
    print('Origin library size: %d'%len(smiles))
    print(smiles[:10])
    chars = []
    with open(voc_fname, 'r') as f:
        for line in f:
            chars.append(line.split()[0])
    print('Vocabulary size: %d'%len(chars))
    print(chars)
    valid_smiles = filter_on_chars(smiles, chars)
    with open(fname, 'w') as f:
        for smiles in valid_smiles:
            f.write(smiles + "\n")

def combine_voc_from_files(fnames):
    """Combine two vocabularies"""
    chars = set()
    for fname in fnames:
        with open(fname, 'r') as f:
            for line in f:
                chars.add(line.split()[0])
    with open("_".join(fnames) + '_combined', 'w') as f:
        for char in chars:
            f.write(char + "\n")

def construct_vocabulary(smiles_list, save_voc):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]

    print("Number of characters: {}".format(len(add_chars)))
    with open(save_voc, 'w') as f:
        for char in add_chars:
            f.write(char + "\n")
    return add_chars

