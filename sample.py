#!/usr/bin/env python
import torch
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys
from data_structs import MolData, Vocabulary
from model import RNN
import os
import sys
import argparse
from utils import valid_smiles
import time

def Sample(model, _voc, num, epoch_num=10000, batch_size=128):
    voc = Vocabulary(init_from_file=_voc)  
    Prior = RNN(voc)
    print(model, num)

    start_time = time.time()

    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(model))
    else:
        Prior.rnn.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    totalsmiles = set()
    enumerate_number = int(num)
    molecules_total = 0
    for epoch in range(1, epoch_num):
        seqs, likelihood, _ = Prior.sample(batch_size)
        valid = 0
        for i, seq in enumerate(seqs.cpu().numpy()):
            smile = voc.decode(seq)
            if valid_smiles(smile) or 1:
                valid += 1
                totalsmiles.add(smile)
                       
        molecules_total = len(totalsmiles)
        print(("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs))))
        print(valid, molecules_total, epoch)
        if molecules_total > enumerate_number:
            break
    return totalsmiles

if __name__ == "__main__":
    model = sys.argv[1]
    _voc = sys.argv[2]
    num = sys.argv[3]
    save = sys.argv[4]
    totalsmiles = Sample(model, _voc, num)
    #f = open('./result/sample_' + os.path.splitext(os.path.split(sys.argv[1])[1])[0] + '_' + str(n) + '.smi', 'w')  
    f = open(save, 'w')
    for smile in totalsmiles:
        f.write(smile + "\n")
    f.close()
    print('Sampling completed')
