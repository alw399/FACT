import torch
import numpy as np
import pandas as pd
from modiscolite.tfmodisco import TFMoDISco
from modiscolite import io
from captum.attr import DeepLiftShap
from cnn import BPNetModel, BPNetK4Model

# --- Utility: Dinucleotide Shuffling ---

def dinuc_shuffle(seq_tensor, num_shuffles=10, seed=None):
    seq_np = seq_tensor.cpu().numpy()
    seq_indices = np.argmax(seq_np, axis=0)
    rng = np.random.default_rng(seed)
    shuffled_list = []
    for _ in range(num_shuffles):
        shuf_indices = _compute_dinuc_shuffle(seq_indices, rng)
        one_hot = np.zeros_like(seq_np)
        for i, idx in enumerate(shuf_indices):
            one_hot[idx, i] = 1.0
        shuffled_list.append(one_hot)
    return torch.tensor(np.stack(shuffled_list), dtype=torch.float32)

def _compute_dinuc_shuffle(seq, rng):
    n = len(seq)
    if n < 2: return seq
    adj = [[] for _ in range(4)]
    for i in range(n - 1):
        adj[seq[i]].append(seq[i+1])
    for v in range(4):
        if adj[v]: rng.shuffle(adj[v])
    new_seq, curr = [seq[0]], seq[0]
    for _ in range(n - 1):
        if not adj[curr]: return seq
        next_nuc = adj[curr].pop(0)
        new_seq.append(next_nuc)
        curr = next_nuc
    return np.array(new_seq)

# --- Attribution Engine ---

class _ProfileSumWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_dual = isinstance(model, BPNetK4Model)
    def forward(self, seq, k4=None):
        if self.is_dual: prof, _ = self.model(seq, k4)
        else: prof, _ = self.model(seq)
        return prof.sum(dim=(1, 2))

class MotifAttributor:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device).eval()
        self.device = device
        self.wrapper = _ProfileSumWrapper(self.model)
        self.dl = DeepLiftShap(self.wrapper)
    def compute(self, x, k4=None, num_baselines=10):
        x_t = x.unsqueeze(0).to(self.device) # (1, 4, L)
        dna_baselines = dinuc_shuffle(x[:4], num_shuffles=num_baselines).to(self.device) # (N, 4, L)
        
        if self.wrapper.is_dual:
            # Ensure K4 is (1, 1, L) to match model expectations and Captum broadcasting
            k4_t = k4.to(self.device)
            while k4_t.dim() < 3:
                k4_t = k4_t.unsqueeze(0)
            
            k4_baselines = k4_t.repeat(num_baselines, 1, 1)
            attr = self.dl.attribute((x_t, k4_t), baselines=(dna_baselines, k4_baselines))
            return attr[0][0].detach().cpu().numpy()
        else:
            attr = self.dl.attribute(x_t, baselines=dna_baselines)
            return attr[0].detach().cpu().numpy()

# --- Fixed Discovery Engine ---

def discover_motifs(model, loader, device='cpu', n_seqs=5000, min_counts=1000, save_path=None):
    """
    Generic motif discovery.
    Note: min_counts reduced to 1000 to capture more SOX2 peaks.
    """
    attributor = MotifAttributor(model, device)
    all_ohs, all_attrs = [], []
    
    print(f"Generating attributions on {device}...")
    count = 0
    
    for batch in loader:
        if count >= n_seqs: break
        x_batch, y_batch = batch[0], batch[-1]
        k4_batch = batch[1] if len(batch) == 3 else None
        
        for i in range(x_batch.shape[0]):
            if count >= n_seqs: break
            if y_batch[i].sum() < min_counts: continue
            
            with torch.enable_grad():
                attr = attributor.compute(x_batch[i], k4_batch[i] if k4_batch is not None else None)
            
            all_ohs.append(x_batch[i][:4].numpy())
            all_attrs.append(attr[:4])
            count += 1
            
                
    if not all_ohs: raise ValueError("No samples passed thresholds.")

    ohs_mod = np.transpose(np.array(all_ohs), (0, 2, 1)).astype(np.float32)
    attrs_mod = np.transpose(np.array(all_attrs), (0, 2, 1)).astype(np.float32)

    print(f"Executing MoDISco-lite on {len(ohs_mod)} sequences...")
    pos_pats, neg_pats = TFMoDISco(
        one_hot=ohs_mod,
        hypothetical_contribs=attrs_mod,
        sliding_window_size=15,
        flank_size=5,
        target_seqlet_fdr=0.15, 
        n_leiden_runs=100,
        verbose=True
    )

    if save_path: io.save_hdf5(save_path, pos_pats, neg_pats, window_size=15)
    return pos_pats, neg_pats