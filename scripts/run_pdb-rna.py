import pandas as pd 
import os 
import shutil 
import argparse
import datetime 

parser = argparse.ArgumentParser()
parser.add_argument("--emb", type=str, help="The name of the desired LLM-dataset combination.", required=True)

args = parser.parse_args()

llm_and_dataset = args.emb.split("_")
llm = llm_and_dataset[0]
dataset = llm_and_dataset[1]
current_timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

df = pd.read_csv(f'data/pdb-rna.csv', index_col="id")
splits = pd.read_csv(f"data/pdb-rna_splits.csv", index_col="id")

train = df.loc[splits.partition=="train"] 
test = df.loc[splits.partition=="test"]
data_path = f"data/pdb_splits/"
out_path = f"results/{current_timestamp}/{dataset}/{llm}"
os.makedirs(data_path, exist_ok=True)
shutil.rmtree(out_path, ignore_errors=True)
os.makedirs(out_path, exist_ok=True)
train.to_csv(f"{data_path}train.csv")
test.to_csv(f"{data_path}test.csv")

os.system(f"python src/train_model.py --emb {args.emb} --train_partition_path {data_path}train.csv --out_path {out_path}")
os.system(f"python src/test_model.py --emb {args.emb} --test_partition_path {data_path}test.csv --out_path {out_path}")
