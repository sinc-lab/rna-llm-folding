import pandas as pd 
import os 
import shutil 
import argparse
from pathlib import Path
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--emb", type=str, help="The name of the desired LLM-dataset combination.", required=True)
parser.add_argument("--weights_path", type=str, help="Path to read model from,  in cases it has to be read from a different place than `results/<embedding>_bpRNA`.")

args = parser.parse_args()

llm_and_dataset = args.emb.split("_")
llm = llm_and_dataset[0]
dataset = llm_and_dataset[1]
current_timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

df = pd.read_csv(f'data/bpRNA.csv', index_col="id")
splits = pd.read_csv(f"data/bpRNA_splits.csv", index_col="id")

train = pd.concat((df.loc[splits.partition=="TR0"], df.loc[splits.partition=="VL0"])) 
test = df.loc[splits.partition=="new"]
data_path = f"data/bprna/"
out_path = f"results/{current_timestamp}/{dataset}_new/{llm}"
os.makedirs(data_path, exist_ok=True)
shutil.rmtree(out_path, ignore_errors=True)
os.makedirs(out_path, exist_ok=True)
train.to_csv(f"{data_path}train.csv")
test.to_csv(f"{data_path}test.csv")


if args.weights_path:
    print(f"skipping bpRNA train, weigths are in {args.weights_path}")
    bprna_weigths_file_path = args.weigths
else:
    bprna_results_path = f"results/{dataset}/{llm}"
    bprna_weigths_file = Path(f"{bprna_results_path}/weights.pmt")
    if bprna_weigths_file.is_file():
        print(f"skipping bpRNA train, weigths are in {bprna_weigths_file}")
        bprna_weigths_file_path = f"{bprna_weigths_file}"
    else:
        os.system(f"python src/train_model.py --emb {args.emb} --train_partition_path {data_path}train.csv --out_path {out_path}")
        bprna_weigths_file_path = f"{out_path}/weigths.pmt"

os.system(f"python src/test_model.py --emb {args.emb} --test_partition_path {data_path}test.csv --out_path {out_path} --weights_path {bprna_weigths_file_path}")
