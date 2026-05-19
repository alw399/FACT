import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import jensenshannon
from motif import MotifAttributor

def load_meme_pwm(meme_path, motif_name="SOX2"):
    """
    Parses a MEME file and returns a (4, L) PyTorch tensor of probabilities for the given motif.
    Columns in JASPAR MEME are A, C, G, T.
    """
    with open(meme_path, 'r') as f:
        lines = f.readlines()
        
    in_matrix = False
    pwm = []
    
    for line in lines:
        if line.startswith("MOTIF"):
            if motif_name in line:
                in_matrix = True
            else:
                in_matrix = False
        elif in_matrix and line.startswith("letter-probability matrix"):
            continue
        elif in_matrix and line.startswith("URL"):
            break
        elif in_matrix:
            # Parse the probability row
            parts = line.strip().split()
            if len(parts) == 4:
                pwm.append([float(x) for x in parts])
                
    if not pwm:
        raise ValueError(f"Motif {motif_name} not found in {meme_path}")
        
    # PWM is now (L, 4), transpose to (4, L)
    return torch.tensor(pwm, dtype=torch.float32).T

def scan_sequences(one_hot_seqs, pwm_prob, pseudocount=1e-4):
    """
    Scans (B, 4, L) one-hot sequences with a (4, w) PWM probability matrix.
    Returns the maximum log-odds score and its start index for each sequence.
    """
    w = pwm_prob.shape[1]
    # Convert probability to log-odds
    pwm_adj = (pwm_prob + pseudocount) / (1.0 + 4 * pseudocount)
    pwm_log = torch.log2(pwm_adj / 0.25) # (4, w)
    
    # Forward strand
    weight_fwd = pwm_log.unsqueeze(0) # (1, 4, w)
    scores_fwd = F.conv1d(one_hot_seqs, weight_fwd).squeeze(1) # (B, L - w + 1)
    
    # Reverse complement strand (flip across 4 channels A<->T, C<->G and flip across length)
    weight_rev = torch.flip(weight_fwd, dims=[1, 2])
    scores_rev = F.conv1d(one_hot_seqs, weight_rev).squeeze(1) # (B, L - w + 1)
    
    # Get max score and pos per sequence
    max_scores_fwd, max_idx_fwd = scores_fwd.max(dim=1)
    max_scores_rev, max_idx_rev = scores_rev.max(dim=1)
    
    # Choose best strand
    use_rev = max_scores_rev > max_scores_fwd
    
    max_scores = torch.where(use_rev, max_scores_rev, max_scores_fwd)
    max_idx = torch.where(use_rev, max_idx_rev, max_idx_fwd)
    
    return max_scores, max_idx, use_rev

def extract_motif_attributions(attributions, max_idx, use_rev, window_size):
    """
    Extracts the attribution scores at the motif hit locations.
    attributions: (B, 4, L)
    max_idx: (B,) start indices
    use_rev: (B,) whether the hit was on the reverse strand
    window_size: int (w)
    
    Returns: (B, 4, w) extracted attributions, reverse-complemented if the hit was on the minus strand
             so they all align in the forward motif orientation.
    """
    B = attributions.shape[0]
    w = window_size
    extracted = torch.zeros(B, 4, w, dtype=attributions.dtype, device=attributions.device)
    
    for i in range(B):
        start = max_idx[i]
        end = start + w
        window_attr = attributions[i, :, start:end]
        
        if use_rev[i]:
            # Reverse complement the attributions (flip bases and spatial dim)
            window_attr = torch.flip(window_attr, dims=[0, 1])
            
        extracted[i] = window_attr
        
    return extracted

def match_motifs_to_attributions(model, loader, meme_path, motif_name="SOX2", device="cpu", head="profile", n_seqs=1000, min_counts=1000, num_baselines=20, jsd_thresh=0.5):
    """
    Runs the targeted motif matching on the dataset.
    Similar to AME, it uses the PWM to find the motif in the sequence,
    but it also extracts the model's attributions at those positions.
    """
    pwm_prob = load_meme_pwm(meme_path, motif_name).to(device)
    window_size = pwm_prob.shape[1]
    
    attributor = MotifAttributor(model, device, head=head)
    
    all_extracted_attrs = []
    all_scores = []
    
    count = 0
    n_skipped_jsd = 0
    print(f"Scanning sequences and extracting attributions for {motif_name}...")
    
    for batch in loader:
        if count >= n_seqs:
            break
            
        x_batch, y_batch = batch[0], batch[-1]
        k4_batch = batch[1] if len(batch) == 3 else None
        
        # We only need the DNA channels (first 4)
        one_hot = x_batch[:, :4].to(device)
        
        # Get log-odds scores and positions
        scores, max_idx, use_rev = scan_sequences(one_hot, pwm_prob)
        
        # Compute predicted profiles for JSD filtering
        with torch.no_grad():
            x_batch_dev = x_batch.to(device)
            k4_batch_dev = k4_batch.to(device) if k4_batch is not None else None
            if k4_batch_dev is not None:
                profile_logits, _ = model(x_batch_dev, k4_batch_dev)
            else:
                profile_logits, _ = model(x_batch_dev)
            
            y_pred_prob = F.softmax(profile_logits.squeeze(1), dim=-1).cpu().numpy()
            target_counts = y_batch.sum(dim=-1)
            y_true_prob = (y_batch / (target_counts.unsqueeze(-1) + 1e-8)).numpy()

        B = x_batch.shape[0]
        for i in range(B):
            if count >= n_seqs:
                break
                
            if y_batch[i].sum() < min_counts:
                continue

            # JSD filter
            jsd = jensenshannon(y_true_prob[i], y_pred_prob[i])
            if np.isnan(jsd) or jsd >= jsd_thresh:
                n_skipped_jsd += 1
                continue
                
            # Compute attributions
            with torch.enable_grad():
                attr = attributor.compute(
                    x_batch[i], 
                    k4_batch[i] if k4_batch is not None else None,
                    num_baselines=num_baselines
                )
                
            attr_tensor = torch.tensor(attr[:4], dtype=torch.float32).to(device).unsqueeze(0) # (1, 4, L)
            
            extracted = extract_motif_attributions(
                attr_tensor, 
                max_idx[i:i+1], 
                use_rev[i:i+1], 
                window_size
            )
            
            all_extracted_attrs.append(extracted[0].cpu().numpy())
            all_scores.append(scores[i].item())
            
            count += 1
            if count % 100 == 0:
                print(f"  Processed {count} sequences")

    print(f"  skipped {n_skipped_jsd} (jsd distance >= {jsd_thresh})")
    
    if not all_extracted_attrs:
        print("No sequences passed thresholds.")
        return None, None
        
    # all_extracted_attrs: (N, 4, w), where N is the number of valid sequences
    return np.array(all_extracted_attrs), np.array(all_scores)
