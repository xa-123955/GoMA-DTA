# GoMA-DTA: A Gene Ontology-Guided Multimodal Attention Fusion Model for Drug–Target Affinity Prediction

## Requirements


Below are the key libraries and their versions:

    python==3.10.0
    torch==2.2.0
    torch-geometric==2.6.1
    torchaudio==2.2.0
    torchvision==0.17.0
    mamba-ssm==2.2.4
    networkx==3.3
    numpy==1.24.3
    rdkit==2024.3.6
    requests==2.32.3
    scikit-learn==1.5.2
    scipy==1.14.1

## Installation

1. create a new conda environment
```  
    conda create --name GoMA-DTA python=3.8

    conda activate GoMA-DTA
```
2. install requried python dependencies
```
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
    pip install torch-geometric==2.6.1
    pip install mamba-ssm networkx numpy rdkit regex requests safetensors scikit-learn scipy setuptools

```

3. Clone the repository:

    git clone https://github.com/xa-123955/GoMA-DTA.git

    cd GoMA-DTA

2. Install the required libraries:

    pip install -r requirements.txt

## Usage

To train the model on your own dataset, follow these steps:

1. Prepare Your Data
   
Make sure your dataset is in the correct format with two columns: drug and target, and corresponding interaction labels.

2. Download Pretrained Models

Download the required large pretrained models from Hugging Face, including:

ESM2(esm2_t30_150M_UR50D)

BlueBERT(bionlpbluebert_pubmed_uncased_L-12_H-768_A-12)

Molformer(MoLFormer-XL-both-10pct)

Place these pretrained model files in the designated folders within the project directory to ensure proper loading during training and inference.

3. Train the Model

For the experiments with vanilla GoMA-DTA, you can directly run the following command. ${dataset} could either be 2016,2020. ${split_task} could be random , time and cluster.

Run the following command to start the training process:

    python main.py --cfg "configs/GoMA-DTA.yaml" --data ${dataset} --split ${split_task}
    
## Directory Structure Description

```
filetree 
├── configs
├── datasets
├── MolFormer
├── ESM2
├── BlueBERT
├── configs.py
├── dataloder.py
├── datapre.py
├── Graph_encoder.py
├── main.py
├── model.py
├── trainer.py
├── util.py
└── README.md

```


## Results

Extensive experiments show that GoMA-DTA outperforms state-of-the-art models on the PDBBind v2016, v2020, and v2024 datasets in both accuracy and robustness, especially under cold-start and challenging conditions. Moreover, the model successfully identifies high-affinity drug candidates targeting the SARS-CoV-2 spike protein, demonstrating its strong practical potential.
