import torch
from rinalmo.pretrained import get_pretrained_model # RiNALMo
import argparse
import pandas as pd
import math
import h5py

parser = argparse.ArgumentParser()

parser.add_argument("--seqs_path", default='./data/ArchiveII.csv', type=str, help="The path of input RNA sequences.")
parser.add_argument("--device", default='cuda:0', type=str, help="Device to execute (either cpu or cuda).")
parser.add_argument("--output_path", default='./data', type=str, help="Folder to save embeddings file.")

args = parser.parse_args()

print("Reading CSV")
data = pd.read_csv(args.seqs_path)

print("Loading RiNALMo model...")
model, alphabet = get_pretrained_model(model_name="giga-v1")
model = model.to(device=args.device)
model.eval()

seqs = data["sequence"].tolist()
seq_ids = data["id"].tolist()
id_to_embedding = {}

print("Generating embeddings...")
for seq_id, seq in zip(seq_ids, seqs):
  # print(f"tokenizing {seq_id}")
  tokens = torch.tensor(alphabet.batch_tokenize([seq]), dtype=torch.int64, device=args.device)

  # print(f"generating {seq_id} repr")
  with torch.no_grad(), torch.cuda.amp.autocast():
    outputs = model(tokens)

  # embedding has size 1 x L x d, so squeeze it first before trimming CLS and END tokens
  id_to_embedding[seq_id] = outputs["representation"].squeeze()[1:-1].to("cpu").numpy()
  # an alternative way of converting it to a h5py file is to convert it to a pandas dataframe first
  # pandas.from_dict() # orient='columns'

print(f"total number of sequences: {len(id_to_embedding)}")

# save in h5 format
print("Saving embeddings...")
with h5py.File(f'{args.output_path}/RiNALMo_ArchiveII.h5', 'w') as hdf:
  for key, value in id_to_embedding.items():
    hdf.create_dataset(key, data=value)
