# ETC

The code repository for the accepted paper of VLDB 24: 'ETC: Efficient Training of Temporal Graph Neural Networks over Large-scale Dynamic Graphs'.

## Requirements
- python >= 3.6.13
- pytorch >= 1.8.1
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.6.1
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0
- openmp >= 201511

Compile C++ temporal sampler (inherited from TGL) first with the following command
> python setup.py build_ext --inplace

## Datasets
We use four datasets in the paper: LASTFM, WIKITALK, STACKOVERFLOW and GDELT.
For LASTFM and GDELT, they can be downloaded from AWS S3 bucket using the `down.sh` script. 
For WIKITALK and STACKOVERFLOW, they can be downloaded from http://snap.stanford.edu/data/wiki-talk-temporal.html and https://snap.stanford.edu/data/sx-stackoverflow.html respectively.
Note that for WIKITALK and STACKOVERFLOW, they need to be preprocessed after obtaining the raw data from the links above. For example:
> python preprocess.py --data \<NameOfDataset> --txt \<PathOfRawData>

## Usage
Example Usage:
> python train.py --data WIKITALK --config config/TGN_WIKITALK.yml --gpu 0
