import paddle
from paddlenlp.transformers import ErnieModel
from tqdm.notebook import tqdm
import h5py

# ========== Set device
paddle.set_device("gpu")
# ========== RNAErnie Model
rna_ernie = ErnieModel.from_pretrained("output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final/")

import pandas as pd
from rna_ernie import BatchConverter
dataset = pd.read_csv("data/ArchiveII.csv", index_col="id")
data = [(k, dataset.loc[k].sequence) for k in dataset.index]

# ========== Batch Converter
batch_converter = BatchConverter(k_mer=1,
                                  vocab_path="./data/vocab/vocab_1MER.txt",
                                  batch_size=64,
                                  max_seq_len=512)


# call batch_converter to convert sequences to batch inputs
emb_dict = {}
for names, _, inputs_ids in tqdm(batch_converter(data)):
    with paddle.no_grad():
        # extract sequence embeddings
        embeddings = rna_ernie(inputs_ids)[0].detach()
        for k in range(embeddings.shape[0]):
            L = len(dataset.loc[names[k]].sequence)
            emb_dict[names[k]] = embeddings[k, 1:L+1, :].cpu().numpy()

with h5py.File(f'RNAErnie_archiveII.h5', 'w') as hdf:
  for key, value in emb_dict.items():
    hdf.create_dataset(key, data=value)