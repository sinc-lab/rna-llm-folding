import os
import argparse
import pandas as pd
import h5py
import torch as tr
import numpy as np

# running in rna-msm conda environment

parser = argparse.ArgumentParser()
parser.add_argument("--seqs_path", default='./data/ArchiveII.csv', type=str, help="The path of input RNA sequences.")
parser.add_argument("--output_path", default='./data', type=str, help="Folder to save embeddings file.")
parser.add_argument("--filename", default='ArchiveII', type=str, help="Datasets ['bpRNA' | 'pdb' | 'ArchiveII'(default)].")
args = parser.parse_args()


#====================================================
def seq2emb(seq_id, seq, current_path):

    seq = seq.upper().replace("T", "U")  # convert to RNA

    # BUILDING TEMPORAL FOLDER
    if not os.path.exists(os.path.join(current_path, 'temp')):
        temp_folder = os.path.join(current_path, 'temp')
        os.mkdir(temp_folder)
    else:
        temp_folder = os.path.join(current_path, 'temp')

    # TXT FILE
    with open(os.path.join(current_path, 'temp.txt'), 'w') as fp:
        fp.write('temp')

    # FASTA FILE
    with open(os.path.join(temp_folder, 'temp.fasta'), 'w') as fp:
        txt = f'>{seq_id}\n{seq}'
        fp.write(txt)

    # A2M_MSA2 FILE (ALIGNMENTS)
    with open(os.path.join(temp_folder, 'temp.a2m_msa2'), 'w') as fp:
        txt = f'>{seq_id}\n{seq}'
        fp.write(txt)

    os.system(f'python RNA_MSM_Inference.py data.root_path={current_path} data.MSA_path={temp_folder} data.model_path={os.path.join(current_path, "pretrained", "RNA_MSM_pretrained.ckpt")} data.MSA_list="temp.txt"')

    emb = np.load(os.path.join(temp_folder, 'temp_emb.npy'))

    if emb.shape[0] != len(seq):
        print(seq_id, emb.shape[0], len(seq))

    return emb
#====================================================


# PATH TO "RNA_MSM_Inference.py" file
current_path = os.path.abspath(args.output_path)

# LOADING INPUT DATA
data = pd.read_csv(args.seqs_path, index_col="id")

id_to_embedding = {}

for i,seq_id in enumerate(data.index):
    print(f'{i+1} / {len(data.index)}')

    seq = data.loc[seq_id, "sequence"]

    embedding = seq2emb(seq_id, seq, current_path)
    id_to_embedding[seq_id] = embedding  #.T # L x C

with h5py.File(f'{args.output_path}/rna-msm_{args.filename}.h5', 'w') as hdf:
  for key, value in id_to_embedding.items():
    hdf.create_dataset(key, data=value)
