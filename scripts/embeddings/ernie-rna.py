import argparse
import pandas as pd
import extract_embedding # ERNIE-RNA
import h5py
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--seqs_path", default='./data/TR0-TS0_4.csv', type=str, help="The path of input RNA sequences.")
parser.add_argument("--device", default='cuda:0', type=str, help="Device to execute (either cpu or cuda).")
parser.add_argument("--output_path", default='./data', type=str, help="Folder to save embeddings file.")

args = parser.parse_args()

print("Reading CSV")
data = pd.read_csv(args.seqs_path)

seqs = data["sequence"].tolist()
seq_ids = data["id"].tolist()
id_to_embedding = {}
count=0


print("Generating embeddings...")
for seq_id, seq in zip(seq_ids, seqs):
  # we rely on the extract_embedding module of ERNIE-RNA to be available. we have to look for another way of invoking it
  embedding = extract_embedding.extract_embedding_of_ernierna(
    [seq],
    if_cls=False,
    arg_overrides={ "data": './src/dict/' },
    pretrained_model_path='./checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt',
    device=args.device,
  )
  print([count, seq_id])
  count = count+1
  # embedding has size 1 x 12 x L x d, so take the output of the last transformer layer
  # and then squeeze it before trimming CLS and END tokens
  id_to_embedding[seq_id] = np.squeeze(embedding[:,11,:,:])[1:-1]
  # an alternative way of converting it to a h5py file is to convert it to a pandas dataframe first
  # pandas.from_dict() # orient='columns'
  
  del embedding
  
    
  
  
print(f"total number of sequences: {len(id_to_embedding)}")

# save in h5 format
print("Saving embeddings...")
with h5py.File(f'{args.output_path}/ERNIE-RNA_bpRNA_4.h5', 'w') as hdf:
  for key, value in id_to_embedding.items():
    hdf.create_dataset(key, data=value)

# h5 file can be created as shown below if a pandas dataframe is chosen
# pandas_dataframe.to_hdf(DATA_PATH + "file.h5", key='key', mode='w')
# h5 file can be later read as this
# rnadist = pd.read_hdf(DATA_PATH + "file.h5")
# instead of using [()] syntax as
# torch.from_numpy(self.embeddings[seq_id][()])
