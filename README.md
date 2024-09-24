# Comprehensive benchmarking on the use of RNA Large Language Models for RNA secondary structure prediction

This repository contains the data and code used in:
    
"Comprehensive benchmarking on the use of RNA Large Language Models for RNA secondary structure prediction," by L.I. Zablocki, L.A. Bugnon, M. Gerard, L. Di Persia, G. Stegmayer, D.H. Milone (under review). Research Institute for Signals, Systems and Computational Intelligence, sinc(i).


In the last three years, a number of RNA-LLM have appeared in literature. We selected them based on their open access availability, and summarized its main features in the table below. See the preprint for details [here](link).

| LLM         | Visualization | Dim | Pre-training sequences | Database   | Architecture (number of layers) | Number of parameters | Repository                                      |
|-------------|----------------------------------------------------------------------------------------------------|---------------------|-----------------------|------------|--------------------------------|----------------------|-------------------------------------------------|
| RNABERT 2022 [[1](https://academic.oup.com/nargab/article/4/1/lqac012/6534363)] |  <img src="fig/rnabert.png" alt="RNABERT"  height = 100px>           | 120           | 70 k              | RNAcentral            | Transformer (6)       | 500 k              | [Link](https://github.com/mana438/RNABERT)       |
| RNA-FM 2022 [[2](https://arxiv.org/abs/2204.00300)]     | <img src="fig/rnafm.png" alt="RNA-FM"  height = 100px>             | 640           | 23 M          | RNAcentral            | Transformer (12)      | 100 M          | [Link](https://github.com/ml4bio/RNA-FM)         |
| RNA-MSM 2024 [[3](https://academic.oup.com/nar/article/52/1/e3/7369930)]  |  <img src="fig/rnamsm.png" alt="RNA-MSM"  height = 100px>           | 768           | 3 M           | Rfam                   | Transformer (12)      | 96 M          | [Link](https://github.com/yikunpku/RNA-MSM)      |
| ERNIE-RNA 2024 [[4](https://www.biorxiv.org/content/10.1101/2024.03.17.585376v1)]  |  <img src="fig/ernierna.png" alt="ERNIE-RNA"  height = 100px>       | 768           | 20 M          | RNAcentral            | Transformer (12)      | 86 M           | [Link](https://github.com/Bruce-ywj/ERNIE-RNA)    |
| RNAErnie 2024 [[5](https://www.nature.com/articles/s42256-024-00836-4)]  |  <img src="fig/rnaernie.png" alt="RNAErnie"  height = 100px>         | 768           | 23 M         | RNAcentral            | Transformer (12)      | 105 M          | [Link](https://zenodo.org/records/10847621)      |
| RiNALMo 2024 [[6](https://arxiv.org/html/2403.00043v1)]    |  <img src="fig/rinalmo.png" alt="RiNALMo"  height = 100px>           | 1280          | 36 M        | RNAcentral +Rfam +Ensembl | Transformer (33) | 650 M          | [Link](https://github.com/lbcb-sci/RiNALMo)       |

## Installation

These steps will guide you through the process of training the secondary structure RNA predictor model, based on the RNA-LLM representations. 
First:
```
git clone https://github.com/sinc-lab/rna-llm-folding
cd rna-llm-comparison/
```
With a conda working installation, run:

```
conda env create -f environment.yml
```
This should install all required dependencies. Then, activate the environment with

```
conda activate rna-llm-folding
```

## RNA-LLM pre-computed embeddings

To train the model, you will need to download the RNA-LLM embedding representations for the desired LLM-dataset combination from the following table, and save them in `/data/embeddings/` directory. 


| ArchiveII   |  bpRNA & bpRNA-new | PDB-RNA |
|-----------|---------|---------|
| one-hot * | [one-hot]()  | [one-hot](https://zenodo.org/api/records/13821093/draft/files/one-hot_pdb-rna.7z/content) |
| [RNABERT](https://zenodo.org/api/records/13821093/draft/files/rnabert_ArchiveII.7z/content)|  [RNABERT](https://zenodo.org/api/records/13821093/draft/files/rnabert_bpRNA.7z/content)|   [RNABERT](https://zenodo.org/api/records/13821093/draft/files/rnabert_pdb-rna.7z/content)|
| [RNA-FM](https://zenodo.org/api/records/13821093/draft/files/rnafm_ArchiveII.7z/content)| [RNA-FM](https://zenodo.org/api/records/13821093/draft/files/rnafm_bpRNA.7z/content)| [RNA-FM](https://zenodo.org/api/records/13821093/draft/files/rnafm_pdb-rna.7z/content)|
| [RNA-MSM](https://zenodo.org/api/records/13821093/draft/files/rna-msm_ArchiveII.7z/content)| [RNA-MSM](https://zenodo.org/api/records/13821093/draft/files/rna-msm_bpRNA.7z/content)| [RNA-MSM](https://zenodo.org/api/records/13821093/draft/files/rna-msm_pdb-rna.7z/content)|
| [ERNIE-RNA](https://zenodo.org/api/records/13821093/draft/files/ERNIE-RNA_ArchiveII.7z/content)| [ERNIE-RNA](https://zenodo.org/api/records/13821093/draft/files/ERNIE-RNA_bpRNA.7z/content)| [ERNIE-RNA](https://zenodo.org/api/records/13821093/draft/files/ERNIE-RNA_pdb-rna.7z/content)|
| [RNAErnie](https://zenodo.org/api/records/13821093/draft/files/RNAErnie_ArchiveII.7z/content)| [RNAErnie](https://zenodo.org/api/records/13821093/draft/files/RNAErnie_bpRNA.7z/content)| [RNAErnie](https://zenodo.org/api/records/13821093/draft/files/RNAErnie_pdb-rna.7z/content)|
| [RiNALMo](https://zenodo.org/api/records/13821093/draft/files/RiNALMo_ArchiveII.7z/content)| [RiNALMo](https://zenodo.org/api/records/13821093/draft/files/RiNALMo_bpRNA.7z/content)| [RiNALMo](https://zenodo.org/api/records/13821093/draft/files/RiNALMo_pdb-rna.7z/content)|

*One-hot embeddings file for the ArchiveII dataset is already provided in the `/data/embeddings` folder.

**Note:** Instructions to generate the RNA-LLM embeddings are detailed in `rna-llm-folding/scripts/embeddings`

## Train and test scripts
Scripts to train and evaluate aRNA-LLM for RNA secondary structure prediction are in the `scripts/` folder. 
Here’s an example: if you wanted to use the one-hot embedding for the ArchiveII dataset (which we provide with this repository), you’d need to run:
```
python scripts/run_archiveII_famfold.py --emb-path one-hot_ArchiveII
```
The --emb option is used to tell the model where to find the embedding representations that will be used to train and test the model. In the example, we used the one-hot embedding for ArchiveII, already available in `/data/embeddings`. By default, the train will be executed on GPU if available. To use other embeddings and datasets, download and place the files from  [here](). 

To run the experiments with other datasets, use “run_bpRNA.py”, “run_bpRNA_new.py” and “run_pdb-rna.py”, which receive the same command line arguments. Results will be saved in `results/{LLM name}_{dataset name}’.

## Comparison results

### Projection of RNA-LLM embeddings
This [notebook](notebooks/UMAP.ipynb) makes use of a UMAP projection technique to illustrate the high-dimensional embeddings into a 3D space.

### Performance on increasing homology challenge datasets
This [notebook](notebooks/violinplots.ipynb) generates the violinplots for performance analysis for each RNA-LLM with the different datasets.
