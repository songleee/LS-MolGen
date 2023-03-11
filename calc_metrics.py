#!/usr/bin/env python
from multiprocessing import Pool
from scipy.spatial.distance import cosine as cos_distance
from utils import mapper, valid_smiles
from utils import disable_rdkit_log, enable_rdkit_log
from utils import read_smiles_csv, read_score_csv, get_ledock_score_parallel
from metrics.utils import compute_fragments, calc_agg_tanimoto, calc_self_tanimoto, \
    compute_scaffolds, fingerprints, canonic_smiles, get_mol
import argparse
import numpy as np
import pandas as pd


def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(valid_smiles, gen)
    return gen.count(True) / len(gen)

def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if
            x is not None]

def fraction_unique(gen, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        canonic.remove(None)
    return len(canonic) / len(gen)

def moses_novelty(gen, train, n_jobs=1):
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return len(gen_smiles_set - train_set) / len(gen_smiles_set)

def scaffold_novelty(gen, train, n_jobs=1):
    # Create the set to store the unique scaffolds
    gen_scaffolds = set(compute_scaffolds(gen, n_jobs=n_jobs))
    train_scaffolds = set(compute_scaffolds(train, n_jobs=n_jobs))

    # Calculate the Scaffold Novelty Score
    scaffold_novelty_score = len(gen_scaffolds - train_scaffolds) / len(gen)

    return scaffold_novelty_score

def scaffold_diversity(gen, n_jobs=1):
    # Create a set to store the unique scaffolds
    scaffolds = compute_scaffolds(gen, n_jobs=n_jobs)

    # Calculate the Scaffold Diversity Score
    scaffold_diversity_score = len(scaffolds) / len(gen)

    return scaffold_diversity_score

def novelty(gen, train, n_jobs=1, device='cpu', fp_type='morgan', gen_fps=None, train_fps=None, p=1):
    """
    Computes novelty, i.e., the diversity between gen and train as:
    1/|A||B| sum_{x, y in AxB} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    if train_fps is None:
        train_fps = fingerprints(train, fp_type=fp_type, n_jobs=n_jobs)
    sim = calc_agg_tanimoto(gen_fps, train_fps, agg='max', device=device, p=p)
    return 1 - np.mean(sim), 1 - np.array(sim)

def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan', gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
#     sim = calc_agg_tanimoto(gen_fps, gen_fps, agg='mean', device=device, p=p)
    sim = calc_self_tanimoto(gen_fps, agg='mean', device=device, p=p)
    return 1 - np.mean(sim), 1 - np.array(sim)

def recovery(gen:list, ref:list):
    """
    Computes recovery rate of ref by gen
    """
    if len(ref) == 0:
        return np.nan
    if len(gen) == 0:
        return 0.0
    _covery = [i for i in ref if i in gen]
    return len(_covery)/len(ref)

def active_rate(score):
    return sum(i<-7 for i in score) / len(score)

def success_rate(gen, train, score, n_jobs=1, device='cuda', fp_type='morgan', gen_fps=None, train_fps=None, p=1):
    """
    Computes success rate, i.e., the rate of generated molecules 
    that satisfy both similarity < 0.7 and docking score < -7 kcal/mol
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    if train_fps is None:
        train_fps = fingerprints(train, fp_type=fp_type, n_jobs=n_jobs)
    sim = calc_agg_tanimoto(gen_fps, train_fps, agg='max', device=device, p=p)
    assert len(sim)==len(score)
    return sum(i<0.7 and j<-7 for i, j in zip(sim, score)) / len(score)


class Metric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pgen, pref)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, pref, pgen):
        raise NotImplementedError

    
class FragMetric(Metric):
    def precalc(self, mols):
        return {'frag': compute_fragments(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return recovery(pgen['frag'], pref['frag'])

class ScafMetric(Metric):
    def precalc(self, mols):
        return {'scaf': compute_scaffolds(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return recovery(pgen['scaf'], pref['scaf'])


def get_all_metrics(gen, n_jobs=1,
                    device='cpu', batch_size=512, pool=None,
                    test=None, train=None, score=None):
    disable_rdkit_log()
    metrics = {}
    # Start the process at the beginning and avoid repeating the process
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    
    metrics['Validity'] = fraction_valid(gen, n_jobs=pool)
    gen_valid = remove_invalid(gen, canonize=True)
    metrics['Uniqueness'] = fraction_unique(gen, pool)
#     div_value, div_distribution = internal_diversity(gen_valid, pool, device=device)
    metrics['Diversity'] = scaffold_diversity(gen_valid, n_jobs=pool)
#     df_distribution = pd.DataFrame({'Smiles': gen_valid, 'Diversity': div_distribution})
    if train is not None:
#         metrics['Moses_Novelty'] = moses_novelty(gen_valid, train, pool)
#         nov_value, nov_distribution = novelty(gen_valid, train, pool, device=device)
        metrics['Novelty'] = scaffold_novelty(gen_valid, train, n_jobs=pool)
#         df_distribution['Novelty'] = nov_distribution
    if test is not None:
        kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
        metrics['Recovery/Frag'] = FragMetric(**kwargs)(gen=gen_valid, ref=test)
        metrics['Recovery/Scaf'] = ScafMetric(**kwargs)(gen=gen_valid, ref=test)
    metrics['Active_rate'] = active_rate(score)
    if train is not None:
        metrics['Success_rate'] = success_rate(gen, train, score, pool, device=device)
    enable_rdkit_log()
    if close_pool:
        pool.close()
        pool.join()
    return metrics#, df_distribution

def main(config):
    gen = read_smiles_csv(config.gen_path)
    train = None
    test = None
    score = None
    dock_file_dir = None
    if config.train_path is not None:
        train = read_smiles_csv(config.train_path)
    if config.test_path is not None:
        test = read_smiles_csv(config.test_path)
    if config.score_path is not None:
        score = read_score_csv(config.score_path)
    else:
        assert config.dock_file_dir is not None
        score = get_ledock_score_parallel(gen, n=config.n_jobs, pool='process', 
                    dock_file_dir=config.dock_file_dir, work_dir=config.dock_file_dir+'_eval_',
                    save_work_dir=False)
        pd.DataFrame({"Smiles": gen, "Ledock": score}).to_csv(config.gen_path, index=None)
    metrics = get_all_metrics(gen=gen, n_jobs=config.n_jobs,
                                 device=config.device,
                                 test=test, train=train, 
                                 score=score)
#     df = pd.read_csv(config.gen_path, usecols=['Smiles', 'Ledock'])
#     df_merge = pd.concat([df, df_distribution], axis=1)
#     df_merge.fillna(0).to_csv(config.gen_path, index=None)
    if config.print_metrics:
        for key, value in metrics.items():
            print('{}, {:.4}'.format(key, value))
    else:
        return metrics

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path',
                        type=str, required=False,
                        help='Path to test molecules csv')
    parser.add_argument('--train_path',
                        type=str, required=False,
                        help='Path to train molecules csv')
    parser.add_argument('--gen_path',
                        type=str, required=True,
                        help='Path to generated molecules csv')
    parser.add_argument('--output',
                        type=str, required=True,
                        help='Path to save results csv')
    parser.add_argument('--print_metrics', action='store_true',
                        help="Print results of metrics or not? [Default: False]")
    parser.add_argument('--score_path',
                        type=str, required=False,
                        help='Path to read ledock score')
    parser.add_argument('--dock_file_dir',
                        type=str, required=False,
                        help='Path to structure file required by ledock')
    parser.add_argument('--n_jobs',
                        type=int, default=1,
                        help='Number of processes to run metrics')
    parser.add_argument('--device',
                        type=str, default='cpu',
                        help='GPU device id (`cpu` or `cuda:n`)')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    metrics = main(config)
    table = pd.DataFrame([metrics]).T
    table.to_csv(config.output, header=False)
