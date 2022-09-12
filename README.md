# DRGN
COLING 2022: Dynamic Relevance Graph Network for Knowledge-Aware Question Answering

## Usage
### Install Dependency files

- Python == 3.7
- Pytorch == 1.12.1
- Transformers == 4.21.3
- Torch-geometric

Run the following commands to create a conda environment (assuming CUDA10.1):
```bash
conda create -n drgn python=3.7
source activate drgn
pip install numpy==1.18.3 tqdm
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers nltk spacy==2.1.6
python -m spacy download en

#for torch-geometric
pip install --upgrade torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu116.html

## config
### The config files:
- data_preprocessing: preprocess.py
- utils: utils/parser_utils.py
- hyper-parameters: qa_drgn.py
- script (CommonsenseQA): drgn_run_script/run_drgn_csqa.sh
- script (OpenbookQA): drgn_run_script/run_drgn_obqa.sh

### Data

Download all the raw data -- ConceptNet, CommonsenseQA, OpenBookQA -- by
```
./download_raw_data.sh
python preprocess.py -p <num_processes>
```
Or
```
./download_preprocessed_data.sh
```

### Training the code:
For CommonsenseQA, run
```
cd drgn_run_script/
sh run_drgn_csqa.sh
```
For OpenBookQA, run
```
cd drgn_run_script/
sh run_drgn_obqa.sh
```