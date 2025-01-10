import pandas as pd 
import os 
import shutil 
import argparse
import datetime
import requests
import py7zr

def download_embedding(emb_name):
  print(f"downloading file {emb_name}")
  response = requests.get(f'https://zenodo.org/records/13821093/files/{emb_name}_ArchiveII.7z?download=1', stream=True)
  response.raise_for_status()

  temp_dir = os.path.join(os.getcwd(), "temp")
  os.makedirs(temp_dir, exist_ok=True)

  print("writing to disk")
  temp_7z_file = os.path.join(temp_dir, "temp.7z")
  with open(temp_7z_file, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024):
      if chunk:
        f.write(chunk)

  print("extracting")
  with py7zr.SevenZipFile(temp_7z_file, "r") as zip_ref:
      zip_ref.extractall(os.getcwd())

  os.remove(temp_7z_file)
  os.rmdir(temp_dir)
  return

# download_embedding("RiNALMo")
# download_embedding("rnafm")
# download_embedding("RNAErnie")
# download_embedding('rna-msm')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ArchiveII", help="The name of the desired LLM-dataset combination.")

args = parser.parse_args()
# rnafm_emb_name = 'rnafm'
# rinalmo_emb_name = 'RiNALMo'
# rnaernie_emb_name = 'RNAErnie'
# rnabert_emb_name = 'rnabert'
# rnamsm_emb_name = 'rna-msm'
# ernierna_emb_name = 'ERNIE-RNA'
# one_hot_emb_name = 'one-hot'

# llm_and_dataset = args.emb.split("_")
# llm = llm_and_dataset[0]
dataset = args.dataset
current_timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')


df = pd.read_csv(f'data/ArchiveII.csv', index_col="id")
splits = pd.read_csv(f'data/ArchiveII_famfold_splits.csv', index_col="id")
# ../data/embeddings
for fam in splits.fold.unique():
    if fam != '5s' and fam != 'srp': # and fam != 'tRNA' and fam != 'RNaseP' and fam != '23s':
        print(f"skipping family {fam}")
        continue
    for llm in ['RiNALMo','rnafm','RNAErnie','rna-msm', 'ERNIE-RNA', 'rnabert']:
      train = df.loc[splits[(splits.fold==fam) & (splits.partition!="test")].index]
      test = df.loc[splits[(splits.fold==fam) & (splits.partition=="test")].index]
      data_path = f"data/archiveII_famfold/{fam}/"
      out_path = f"results/revision/100epochs/lr1e-3/{dataset}_famfold/{llm}/{fam}"
      os.makedirs(data_path, exist_ok=True)
      shutil.rmtree(out_path, ignore_errors=True)
      os.makedirs(out_path, exist_ok=True)
      train.to_csv(f"{data_path}train.csv")
      test.to_csv(f"{data_path}test.csv")

      print("+" * 80)
      print(f"ArchiveII {fam} TRAINING STARTED, llm {llm}, dataset {dataset}".center(80))
      print("+" * 80)
      os.system(f"python src/train_model.py --emb {llm}_{dataset} --train_partition_path {data_path}train.csv --val_partition_path {data_path}test.csv --out_path {out_path} --max_epochs 100 --lr 1e-3")
      print(f"ArchiveII {fam} TRAINING ENDED".center(80))
      # # print("+" * 80)
      # # print(f"ArchiveII {fam} TESTING STARTED".center(80))
      # # print("+" * 80)
      # # os.system(f"python src/test_model.py --emb {args.emb} --test_partition_path {data_path}test.csv --out_path {out_path}")
      # print(f"ArchiveII {fam} TESTING ENDED".center(80))
