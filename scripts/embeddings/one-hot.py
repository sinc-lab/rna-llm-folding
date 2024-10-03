"""Script to generate embeddings using one-hot encoding."""
import argparse
import pandas as pd
import h5py
import torch as tr

parser = argparse.ArgumentParser()
# Inputs: archiveII/TR0-TS0 from https://github.com/sinc-lab/sincFold/tree/main/data
parser.add_argument("--seqs_path", default='./data/ArchiveII.csv', type=str, help="The path of input RNA sequences.")
parser.add_argument("--device", default='cuda:0', type=str, help="Device to execute (either cpu or cuda).")
parser.add_argument("--output_path", default='./data', type=str, help="Folder to save embeddings file.")
args = parser.parse_args()

# esto es solo para este tipo de embedding, cada uno debería usar el LLM que tocó
NT_DICT = {
    "R": ["G", "A"],
    "Y": ["C", "U"],
    "K": ["G", "U"],
    "M": ["A", "C"],
    "S": ["G", "C"],
    "W": ["A", "U"],
    "B": ["G", "U", "C"],
    "D": ["G", "A", "U"],
    "H": ["A", "C", "U"],
    "V": ["G", "C", "A"],
    "N": ["G", "A", "C", "U"],
}
VOCABULARY = ["A", "C", "G", "U"]
def seq2emb(seq, pad_token="-"):
    """One-hot representation of seq nt in vocabulary.  Emb is CxL
    Other nt are mapped as shared activations.
    """
    seq = seq.upper().replace("T", "U")  # convert to RNA
    emb_size = len(VOCABULARY)
    emb = tr.zeros((emb_size, len(seq)), dtype=tr.float)

    for k, nt in enumerate(seq):
        if nt == pad_token:
            continue
        if nt in VOCABULARY:
            emb[VOCABULARY.index(nt), k] = 1
        elif nt in NT_DICT:
            v = 1 / len(NT_DICT[nt])
            ind = [VOCABULARY.index(n) for n in NT_DICT[nt]]
            emb[ind, k] = v
        else:
            raise ValueError(f"Unrecognized nucleotide {nt}")

    return emb

# carga datos de entrada
data = pd.read_csv(args.seqs_path, index_col="id")

id_to_embedding = {}
# generate dictionary with seq ids as keys, and embedding tensors as values
for seq_id in data.index:
  seq = data.loc[seq_id, "sequence"]

  # on-hot encoding of RNA sequences. ojo que de que no guarde el gradiente (hacer un detach() o tr.no_grad()). 
  # No es necesario hacer padding, cada embedding es del mismo tamaño que la secuencia.  
  embedding = seq2emb(seq) 
  id_to_embedding[seq_id] = embedding.T # L x C
  
# h5, parquet, pickle, npy are possible file formats to store the representations
# we choose h5 here
with h5py.File(f'{args.output_path}/one-hot_archiveII.h5', 'w') as hdf:
  for key, value in id_to_embedding.items():
    hdf.create_dataset(key, data=value)
