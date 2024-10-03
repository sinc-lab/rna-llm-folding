"""Script to generate embeddings using RNABERT"""
from tqdm import tqdm
import h5py
import ast
import os
import numpy as np
import pandas as pd
# running in rnabert conda environment ONLY FOR SEQ LENGTH L<440

#export PRED_FILE=sample/aln/sample.raw.fa
#export PRE_WEIGHT=bert_mul_2.pth
# example: 
#   python3 MLM_SFP.py --pretraining ${PRE_WEIGHT} --data_embedding ${PRED_FILE} --embedding_output ${OUTPUT_FILE} --batch 40 

# generate embeddings using RNABERT
def seq2emb(seq, seq_id):
  if len(seq)<440:
    file = open('data/input.fa', 'w')
    file.write('> rna \n' )
    file.write(seq)
    file.close()

    os.system("python RNABERT/MLM_SFP.py --pretraining bert_mul_2.pth --data_embedding data/input.fa --embedding_output output/embedding.txt --batch 40")
    try:
      with open("output/embedding.txt", 'r') as f: lines = f.readlines()
    except FileNotFoundError:
      print("File does not exist:", seq_id)
    emb = np.asarray(ast.literal_eval(lines[0]))
    os.remove("output/embedding.txt")
  else:
    file = open('data/input1.fa', 'w')
    file.write('> rna \n' )
    file.write(seq[0:440-1])
    file.close()
    os.system("python RNABERT/MLM_SFP.py --pretraining bert_mul_2.pth --data_embedding data/input1.fa --embedding_output output/embedding1.txt --batch 40")
    try:
      with open("output/embedding1.txt", 'r') as f: lines = f.readlines()
    except FileNotFoundError:
      print("File does not exist:", seq_id)
    emb1 = np.asarray(ast.literal_eval(lines[0]))
    print(emb1.shape)
    os.remove("output/embedding1.txt")
    
    file = open('data/input2.fa', 'w')
    file.write('> rna \n' )
    file.write(seq[440-64-1:])
    file.close()
    os.system("python RNABERT/MLM_SFP.py --pretraining bert_mul_2.pth --data_embedding data/input2.fa --embedding_output output/embedding2.txt --batch 40")
    try:
      with open("output/embedding2.txt", 'r') as f: lines = f.readlines()
    except FileNotFoundError:
      print("File does not exist:", seq_id)
    emb2 = np.asarray(ast.literal_eval(lines[0]))
    print(emb2.shape)
    os.remove("output/embedding2.txt")

    emb = np.concatenate((emb1[0:440-32-1,:],emb2[32:,:]), axis=0)

  return emb

# load data
data = pd.read_csv("ArchiveII.csv", index_col="id")
#data = pd.read_csv("TR0-TS0.csv", index_col="id")


# generate a dictionary with seq ids as keys, and embedding tensors as values
print("Generating embeddings...")
id_to_embedding = {}
for seq_id in tqdm(data.index):
  print(seq_id)
  seq = data.loc[seq_id, "sequence"]
  embedding = seq2emb(seq, seq_id)
  id_to_embedding[seq_id] = embedding


  
# save in h5 format
print("Saving embeddings...")
with h5py.File('rnabert_ArchiveII.h5', 'w') as hdf:
  for key, value in tqdm(id_to_embedding.items()):
    hdf.create_dataset(key, data=value)
    