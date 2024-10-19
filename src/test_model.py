import numpy as np
import argparse
import torch
import pandas as pd
import logging
import os

from model import SecondaryStructurePredictor
from dataset import create_dataloader
from utils import get_embed_dim

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Applied workaround for CuDNN issue, install nvrtc.so
# Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR

parser = argparse.ArgumentParser()

parser.add_argument("--emb", type=str, help="The name of the desired LLM-dataset combination.")
parser.add_argument("--test_partition_path", type=str, help="The path of the test partition.")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size to use in forward pass.")
parser.add_argument("--out_path", default="results", type=str, help="Path to read model from, and to write predictions/metrics/logs")
parser.add_argument("--weights_path", type=str, help="Path to read model from, in cases it has to be read from a different place than `out_path`")

args = parser.parse_args()

if torch.cuda.is_available():
    device=f"cuda:{torch.cuda.current_device()}"
else:
    device='cpu'

# Create results file with the name of input file
out_name = os.path.splitext(os.path.split(args.test_partition_path)[-1])[0]

embeddings_path = f"data/embeddings/{args.emb}.h5"

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join(args.out_path, f'log-{out_name}.txt'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Run on {args.out_path}, with device {device} and embeddings {embeddings_path}")
logger.info(f"Testing with file: {args.test_partition_path}")

test_loader = create_dataloader(
    embeddings_path,
    args.test_partition_path,
    args.batch_size,
    False
)
embed_dim = get_embed_dim(test_loader)
best_model = SecondaryStructurePredictor(embed_dim=embed_dim, device=device)
best_model.load_state_dict(torch.load(args.weights_path if args.weights_path else os.path.join(args.out_path, f"weights.pmt"), map_location=torch.device(best_model.device)))
best_model.eval()


metrics = best_model.test(test_loader)
metrics = {f"test_{k}": v for k, v in metrics.items()}
logger.info(" ".join([f"{k}: {v:.3f}" for k, v in metrics.items()]))

out_file = os.path.join(args.out_path, f"metrics_{out_name}.csv")
pd.set_option('display.float_format','{:.3f}'.format)
pd.DataFrame([metrics]).to_csv(out_file, index=False)

out_file = os.path.join(args.out_path, f"preds_{out_name}.csv")
predictions = best_model.pred(test_loader)
predictions.to_csv(out_file, index=False)
