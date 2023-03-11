#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from tqdm import tqdm
import data_structs as ds
from data_structs import MolData, Vocabulary
from utils import Variable, decrease_learning_rate
from model import RNN
import math

def Transfer(smiles_file='data/DRD3.smi', voc_file='data/Voc', restore_from='data/Prior.ckpt',
             batch_size=16, learning_rate=0.001, epoch_num=100,
             lr_decrease_rate=0.03, save_dir='data/Transfer.ckpt',
             early_stop=20):
    """Trains the Prior RNN"""
#    ds.filter_file_on_chars(smiles_file, voc_file, smiles_file)
    voc = Vocabulary(init_from_file=voc_file)
    print("Reading smiles...")
    smiles_list = ds.canonicalize_smiles_from_file(smiles_file)
    moldata = MolData(smiles_list, voc)
    data = DataLoader(moldata, batch_size=batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)
    
    Prior = RNN(voc)
    
    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr=learning_rate)
    
    best_loss, early_stop_count = math.inf, 0
    for epoch in range(1, epoch_num+1):
        loss_record = []
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
            loss_record.append(loss.detach().item())

            # Every 2 epoch we decrease learning rate and print some information
            if epoch % 2 == 0 and step == 1:
                decrease_learning_rate(optimizer, decrease_by=lr_decrease_rate)
            if epoch % 10 == 0 and step == 1:
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.item()))
                seqs, likelihood, _ = Prior.sample(100)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 10:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")               
        mean_train_loss = sum(loss_record)/len(loss_record)
        tqdm.write(f'Epoch [{epoch}/{epoch_num}]: Train loss: {mean_train_loss:.4f}')
        # Save the model
        if mean_train_loss < best_loss:
            best_loss = mean_train_loss
            torch.save(Prior.rnn.state_dict(), save_dir)
            tqdm.write('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= early_stop:
            tqdm.write('\nModel is not improving, so we halt the training session.')
            return

if __name__ == "__main__":
    Transfer(smiles_file='data/Mpro/Mpro_chembl_article.smi', restore_from='data/Mpro/Prior.ckpt',
             batch_size=1, learning_rate=0.001, voc_file='data/Mpro/Voc',
             epoch_num=500, lr_decrease_rate=0.03,
             early_stop=20, save_dir='data/Mpro/Transfer.ckpt')

