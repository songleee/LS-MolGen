#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm
import data_structs as ds
from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')

def pretrain(smiles_file='data/chembl31.smi', save_voc='data/Voc', restore_from=None,
             batch_size=128, learning_rate=0.001, epoch_num=5, 
             lr_decrease_rate=0.03, save_dir='data/Prior.ckpt'):
    """Trains the Prior RNN"""

    print("Reading smiles...")
    smiles_list = ds.canonicalize_smiles_from_file(smiles_file)
    # Read vocabulary from a file or create a new one
    print("Constructing vocabulary...")
    ds.construct_vocabulary(smiles_list, save_voc)
    voc = Vocabulary(init_from_file=save_voc)

    # Create a Dataset from a SMILES list
    moldata = MolData(smiles_list, voc)
    data = DataLoader(moldata, batch_size=batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = RNN(voc)

    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = learning_rate)
    for epoch in range(1, epoch_num+1):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=lr_decrease_rate)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.item()))
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                #torch.save(Prior.rnn.state_dict(), "data/Prior.ckpt")

        # Save the Prior
        torch.save(Prior.rnn.state_dict(), save_dir)

if __name__ == "__main__":
    pretrain(smiles_file='data/chembl31.smi', save_voc='data/Voc',
             batch_size=128, learning_rate=0.001,
             epoch_num=5, lr_decrease_rate=0.03,
             save_dir='data/Prior.ckpt')

