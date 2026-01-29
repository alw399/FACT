# %%
# %%
import sys 
sys.path.append('../src')

# %%
from cnn_k4 import *
from cnn import *
from datas import *
import os

bigwig_path = '../data/bws_for_predictor/SPT16_4D_6HdTAG_Sox2_S49_1_120_sorted.bw'
k4_bigwig_path = '../data/bws_for_predictor/SPT16_4D_6HdTAG_K4_S55_150_500_scaled.bw'
timepoint = os.path.basename(bigwig_path).split('_')[2]

fasta_path = '../data/GRCm38.primary_assembly.genome.fa'
chrom = 'chr3'
window_size = 1000
stride = 1000
signal_bins = 100
batch_size = 256
epochs = 200
lr = 1e-3
device = 'cuda'
weight_decay = 1e-4
patience = 10

dataset = SequenceDualBigWigDataset(
    fasta_path=str(fasta_path),
    k4_bigwig_path=str(k4_bigwig_path),
    target_bigwig_path=str(bigwig_path),
    signal_bins=signal_bins,
    chrom=chrom,
    window_size=window_size,
    stride=stride
)

config = TrainConfig(
    lr=lr,
    weight_decay=weight_decay,
    patience=patience,
    epochs=epochs,
    device=device,
)

model = train_bpnet_k4(
    dataset=dataset,
    batch_size=batch_size,
    config=config
)

import time
from pathlib import Path

date = time.strftime("%Y%m%d_%H")
output_model_path = Path(f'../models/seqk4_{timepoint}_{date}.pt')

save_model(model, output_model_path)
print(f"Saved trained model (config + state_dict) to: {output_model_path}")

visualize_split_predictions(
    model,
    device='cuda',
    n_examples_per_split=10,
    save_path=f'../models/results/seqk4_{timepoint}_{date}'
)


