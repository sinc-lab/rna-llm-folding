import pandas as pd
import os 
import shutil 
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--emb", type=str, default="one-hot_ArchiveII", help="The name of the desired LLM-dataset combination.")

args = parser.parse_args()

llm_and_dataset = args.emb.split("_")
llm = llm_and_dataset[0]
dataset = llm_and_dataset[1]
current_timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

df = pd.read_csv(f'data/ArchiveII.csv', index_col="id")
splits = pd.read_csv(f'data/ArchiveII_kfold_splits.csv', index_col="id")

for k in range(5):
    train = df.loc[splits[(splits.fold==k) & (splits.partition!="test")].index]
    test = df.loc[splits[(splits.fold==k) & (splits.partition=="test")].index]
    data_path = f"data/archiveII_kfold/{k}/"
    out_path = f"results/{dataset}_kfold_{llm}_{k}_{current_timestamp}"
    os.makedirs(data_path, exist_ok=True)
    shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path, exist_ok=True)
    train.to_csv(f"{data_path}train.csv")
    test.to_csv(f"{data_path}test.csv")

    os.system(f"python src/train_model.py --emb {args.emb} --train_partition_path {data_path}train.csv --out_path {out_path}")
    os.system(f"python src/test_model.py --emb {args.emb} --test_partition_path {data_path}test.csv --out_path {out_path}")
