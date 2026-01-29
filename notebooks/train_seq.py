import sys 
sys.path.append('../src')

from cnn import * 
from datas import * 
import os

# bigwig_path = f'../data/bws_for_predictor/SPT16_4D_2HdTAG_Sox2_S46_1_120_sorted.bw'
bigwig_path = '../data/bws_for_predictor/SPT16_4D_6HdTAG_Sox2_S49_1_120_sorted.bw'
# bigwig_path = f'../data/bws_for_predictor/SPT16_4D_10dTAG_Sox2_rep2_S58_1_120_sorted.bw'
# bigwig_path = f'../data/bws_for_predictor/SPT16_4D_DMSO_Sox2_S40_1_120_sorted.bw'
# bigwig_path = '../data/bws_for_predictor//ix/djishnu/alw399/FACT/data/bws_for_predictor/SPT16_D4_30dTAG_Sox2_S7_1_120_sorted.bw'
timepoint = os.path.basename(bigwig_path).split('_')[2]

fasta_path = '../data/GRCm38.primary_assembly.genome.fa'
chrom = 'chr3'
window_size = 1000
stride = 1000
signal_bins = 100
batch_size = 256
epochs = 200
patience=20
lr = 1e-3
device = 'cuda'
weight_decay = 1e-4

dataset = SequenceBigWigDataset(
    fasta_path=str(fasta_path),
    bigwig_path=str(bigwig_path),
    signal_bins=signal_bins,
    chrom=chrom,
    window_size=window_size,
    stride=stride
)

config = TrainConfig(
    lr=lr,
    weight_decay=weight_decay,
    epochs=epochs,
    device=device,
    patience=patience
)

model = train_cnn_regressor(
    dataset=dataset,
    batch_size=batch_size,
    config=config,
)

import time
from pathlib import Path

date = time.strftime("%Y%m%d_%H")
output_model_path = Path(f'../models/seq_{timepoint}_{date}.pt')

save_model(model, output_model_path)
print(f"Saved trained model (config + state_dict) to: {output_model_path}")


visualize_split_predictions(
    model,
    device=device,
    n_examples_per_split=10,
    save_path=f'../models/results/seqk4_{timepoint}_{date}'
)