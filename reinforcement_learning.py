#!/usr/bin/env python
import torch
import numpy as np
import argparse
import warnings
import time
import os
from model import RNN
import data_structs as ds
from data_structs import Vocabulary, Inception
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique
from utils import prepare_receptor, get_ledock_score_parallel
import math
from rdkit import Chem, rdBase
from rdkit.Chem import QED
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

def train_agent(restore_prior_from='data/Transfer.ckpt',
                restore_agent_from='data/Transfer.ckpt',
                save_dir='./data', voc_file='data/Voc',
                learning_rate=0.0005,
                batch_size=128, n_steps=100,
                num_processes=16, pdb=None, dock_file_dir='data/ledock',
                sigma=20, experience_replay=0, early_stop=20):

    voc = Vocabulary(init_from_file=voc_file)

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)
    
    if not os.path.exists(dock_file_dir):
        prepare_receptor(pdb, dock_file_dir, rmsd=1.0, binding_pose_num=5)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_prior_from))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=learning_rate)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Inception(voc)

    # Information for the logger
    step_score = [[], []]

    print("Model initialized, starting training...")

    best_loss, early_stop_count = math.inf, 0
    for step in range(n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc)
        score = scoring_function(smiles, num_processes=num_processes, dock_file_dir=dock_file_dir, k=-10)
        score, scaffold, scaf_fp = experience.update_score(smiles, score)

        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        # loss = - agent_likelihood * Variable(score)

        # Experience Replay
        # First sample
        if experience_replay and len(experience)>4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            # exp_loss = - exp_agent_likelihood * Variable(exp_score)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        experience.add_experience(smiles, score, prior_likelihood, scaffold, scaf_fp)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        # loss_p = - (1 / agent_likelihood).mean()
        # loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_train_loss = loss.detach().item()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
                step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(10):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                           prior_likelihood[i],
                                                                           augmented_likelihood[i],
                                                                           score[i],
                                                                           smiles[i]))
        # Need this for Vizard plotting
        step_score[0].append(step + 1)
        step_score[1].append(np.mean(score))
        print(f'Step [{step+1}/{n_steps}]: Train loss: {mean_train_loss:.4f}')
        # Save the model
        if mean_train_loss < best_loss:
            best_loss = mean_train_loss
            torch.save(Prior.rnn.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= early_stop:
            print('\nModel is not improving, so we halt the training session.')
            break

    # Save files
    experience.save_memory(os.path.join(save_dir, "memory.csv"))
    with open(os.path.join(save_dir, 'step_score.csv'), 'w') as f:
        f.write("step,score\n")
        for s1, s2 in zip(step_score[0], step_score[1]):
            f.write(str(s1) + ',' + str(s2) + "\n")

def scoring_function(smiles, dock_file_dir, k=-10, num_processes=16):
    docking_score = get_ledock_score_parallel(smiles, n=num_processes, pool='process', 
                    dock_file_dir=dock_file_dir, work_dir=dock_file_dir+'_', 
                    save_work_dir=False)
    docking_score = [max(_score, k) / k for _score in docking_score]
    qed_score = [QED.qed(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) is not None else 0 for smi in smiles]
    score = [_dock if _qed >= 0.34 else 0 for _dock, _qed in zip(docking_score, qed_score)]
    return np.array(score, dtype=np.float32)


if __name__ == "__main__":
    s = time.time()
    train_agent(restore_prior_from='data/EGFR/Transfer.ckpt',
                restore_agent_from='data/EGFR/Prior.ckpt',
                save_dir='data/EGFR', voc_file='data/EGFR/Voc',
                learning_rate=0.0005,
                batch_size=128, n_steps=3000,
                num_processes=32, pdb='data/EGFR/EGFR_2RGP.pdb', dock_file_dir='data/EGFR/ledock',
                sigma=128, experience_replay=1, early_stop=200)
    e = time.time()
    print("Use time: {:.4f}s".format(e - s))
