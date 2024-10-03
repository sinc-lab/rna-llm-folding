"""Script to generate embeddings using RNA-FM."""
import argparse
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
import torch as tr
import fm             # running in RNA-FM conda environment

parser = argparse.ArgumentParser()
# inputs: archiveII/TR0-TS0 from https://github.com/sinc-lab/sincFold/tree/main/data
# example: 
#   python3 rnafm.py --seqs_path /DATA/lncRNA/data/sincfold/ArchiveII.csv --device cuda:0 --output_path /DATA/lncRNA/results/ 
parser.add_argument("--seqs_path", default='./data/ArchiveII.csv', type=str, help="The path of input RNA sequences.")
parser.add_argument("--device", default='cuda:0', type=str, help="Device to execute (either cpu or cuda).")
parser.add_argument("--output_path", default='./data', type=str, help="Folder to save embeddings file.")
args = parser.parse_args()

# generate embeddings using RNA-FM
def seq2emb(seq, model, batch_converter, device):
  data = [("RNA1", seq)]
  batch_labels, batch_strs, batch_tokens = batch_converter(data)
  # print("batch_tokens", batch_tokens) # includes additional tokens for start and end
  with tr.no_grad():
    results = model(batch_tokens.to(device), repr_layers=[12])    
  emb = results["representations"][12]
  return emb

# load data
data = pd.read_csv(args.seqs_path, index_col="id")

# load RNA-FM model
print("Loading RNA-FM model...")
device = tr.device(args.device)
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)
model.eval()  # disables dropout for deterministic results

# generate a dictionary with seq ids as keys, and embedding tensors as values
print("Generating embeddings...")
id_to_embedding = {}
for seq_id in tqdm(data.index):
  seq = data.loc[seq_id, "sequence"]
  embedding = seq2emb(seq, model, batch_converter, device)
  #id_to_embedding[seq_id] = embedding.squeeze().to("cpu").numpy()
  # remove the first and last tokens (start and end)
  id_to_embedding[seq_id] = embedding.squeeze()[1:-1].to("cpu").numpy()
  # print(seq_id, "\n", id_to_embedding[seq_id])
  # print(seq_id, len(data.loc[seq_id, "sequence"]), id_to_embedding[seq_id].shape)
  
# save in h5 format
print("Saving embeddings...")
with h5py.File(f'{args.output_path}/rnafm_ArchiveII.h5', 'w') as hdf:
  for key, value in tqdm(id_to_embedding.items()):
    hdf.create_dataset(key, data=value)
