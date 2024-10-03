# One-hot
Nothing to install here :)

# ERNIE-RNA

```
git clone https://github.com/Bruce-ywj/ERNIE-RNA.git
```

Download pretrained weights from https://drive.google.com/drive/folders/1iX-xtrTtT-zk5je8hCdYQOQHDWbl1wgo and put them under `ERNIE-RNA/checkpoint/ERNIE-RNA_checkpoint`.

```
cd ./ERNIE-RNA
conda env create -f environment.yml
conda activate ERNIE-RNA
conda install pandas
conda install h5py
python ernie-rna.py
```

# RiNALMo
```
git clone https://github.com/lbcb-sci/RiNALMo.git
cd ./RiNALMo
conda create -n "RiNALMo" python=3.11
conda activate RiNALMo
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install .
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation
conda install pandas
conda install h5py
python rinalmo.py
```

# RNA-FM
```
git clone https://github.com/ml4bio/RNA-FM.git
cd RNA-FM/
conda env create -f environment.yml
conda activate RNA-FM
cd redevelop/
pip install .
python3 rnafm.py
```

# RNABERT
```
conda create -n rnabert python=3.6.5
conda activate rnabert
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install attrdict matplotlib sklearn gdown biopython==1.76

git clone https://github.com/mana438/RNABERT
cd RNABERT/
python setup.py install
gdown --id 1sT6jlv9vrpX0npKmnbFeOqZ1JZDrZTQ2 --output bertrna_full.zip
unzip bertrna_full.zip
export PRED_FILE=sample/aln/sample.raw.fa
export PRE_WEIGHT=bert_mul_2.pth

python rnabert.py
```

### WARNING
restriction for L<440, seq must be all in uppercase, special nts are not recognized

# RNAErnie

Follow the instructions form the original repository. If that not work, install the following packages manually: 
```
pip install paddlepaddle-gpu
pip install biopython
pip install paddlenlp
```
Install the repository and model weights
```
git clone https://github.com/CatIIIIIIII/RNAErnie.git
gdown 1MQjtnrtssoF5qAiALakaDDBnQfy0PJC4
gdown 1Cknx2StFQAm-aQtvDpFud5Ii1qlTCVa8
mv model_config.json model_state.pdparams output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final/

python RNAERNIE.py
```

# RNA-MSM

## Conda environment creation
```
conda create -n rnamsm python=3.8.5
conda activate rnamsm
```

## Required packages
```
conda install hhsuite=3.3.0
pip install tqdm pytorch_lightning transformers msm scipy Bio tape_proteins scikit-learn numba hydra hydra-core

git clone git@github.com:yikunpku/RNA-MSM.git
cd ./RNA-MSM
```

## Required modifications to run RNA-MSM code
```
> utils/_init_.py
    - REPLACE:
        #from .version import version as _version_  # noqa
    - BY:
        _version_ = '0.1.0'

    - COMMENT LINE 10 --> #from . import pretrained  # noqa
```

```
> RNA_MSM_Inference.py
    - REPLACE:
        def extract_feat(cfg: Config) -> None:
    - BY:
        @hydra.main(config_name="config", version_base="1.2") -> None:
```

```
> utils/align.py
    - ADD "import os" en línea 1

    - COMMENT:
        ".split()" en la línea "command = " ".join(..."

    - REPLACE:
        result = subprocess.run(command, capture_output=True)
        result.check_returncode()
    - POR:
        os.system(command)
```

## TO run RNA-MSM inference

```
python RNA_MSM_Inference.py data.root_path='<FULLPATH TO RNA-MSM>' \
                            data.MSA_path='<FULLPATH TO RNA-MSM>/results' \
                            data.model_path='<FULLPATH TO RNA-MSM>/pretrained/RNA_MSM_pretrained.ckpt' \
                            data.MSA_list=rna_id.txt
```

## Notes about RNA-MSM
- An IDX file can be created for each experiment, with the list of sequence names to be analyzed.
- Each sequence must have its ".fasta" file and its ".a2m_msa2" file (alignments).
