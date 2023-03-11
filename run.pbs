#!/bin/bash
#PBS -N run_LS-MolGen
#PBS -e lsmolgen.out
#PBS -o lsmolgen.out
#PBS -q ai
#PBS -l select=1:ncpus=32:ngpus=1:mem=16gb
#PBS -l walltime=72:00:00

cd $PBS_O_WORKDIR
#module purge
#module load Anaconda3
export PATH=/home/lisong/software/anaconda3/bin:$PATH
export PATH=/home/lisong/local/bin:$PATH
export PATH=/home/lisong/software/openbabel/bin:/home/lisong/software/LeDOCK:$PATH
source activate LS-MolGen

#python pre_train.py > prior.log
#python transfer_learning.py > tl.log
python reinforcement_learning.py > data/EGFR/reinvent/rl.log

