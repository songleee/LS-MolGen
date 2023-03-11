import sys
import data_structs as ds


if __name__ == "__main__":
    smiles_file = sys.argv[1]
    #smiles_file = "data/prior_trainingset"
    print("Reading smiles...")
    smiles_list = ds.canonicalize_smiles_from_file(smiles_file)
    print("Constructing vocabulary...")
    voc_chars = ds.construct_vocabulary(smiles_list)
    fname = sys.argv[2]
    #fname = "data/mols_filtered.smi"
    ds.write_smiles_to_file(smiles_list, fname)