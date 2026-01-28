import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_transformer_flops(seq_len, d_model, n_heads, n_layers, method='full', sparsity=0.1, window_size=128):
    """
    Calculate theoretical FLOPs for one forward pass.
    Simplified estimation focusing on Attention and FFN.
    """
    # Common parts
    # 1. Projections (Q, K, V, O): 4 * N * d^2
    proj_flops = 4 * seq_len * (d_model ** 2)
    
    # 2. FFN (up proj, down proj, act): 
    # Usually 2 linear layers: d -> 4d -> d
    # 2 * N * d * 4d = 8 * N * d^2
    ffn_flops = 8 * seq_len * (d_model ** 2)
    
    # Attention Interaction
    if method == 'full':
        # Q * K^T: N * N * d
        # Attn * V: N * N * d
        # Total: 2 * N^2 * d
        attn_ops = 2 * (seq_len ** 2) * d_model
    elif method == 'sparse' or method == 'spdg':
        # SPDG/Fixed uses Window + Selected Indices
        # Effective attention span = sparsity * N (if percent) or Fixed Window?
        # Assuming 'sparsity' parameter controls the number of attended tokens per query.
        # Let's assume K = sparsity * N (for fractional sparsity) or min(N, window_size)
        
        # Here we model linear complexity: O(N * K)
        # K is average number of attended tokens.
        # For SPDG, K varies, but we aim for sparsity ~ 0.1 (10%)
        # But if sparsity is fixed window (e.g. 128), it's O(N * 128).
        # Let's assume K = window_size
        k_tokens = window_size # Linear scaling
        # Or mixed?
        # Let's use the standard "Sparsity" percentage for fair "Compression" comparison.
        # If sparsity = 0.1, it means we calculate 10% of N^2.
        
        # However, academic claim is O(N).
        # Let's assume Fixed Sparse / SPDG uses a local window of size 'window_size'.
        # And maybe some random global tokens.
        # Let's use K = window_size + global_tokens (~32)
        k_effective = window_size + 32
        
        attn_ops = 2 * seq_len * k_effective * d_model
    
    # Per layer
    layer_flops = proj_flops + ffn_flops + attn_ops
    
    total_flops = n_layers * layer_flops
    
    # SPDG has extra Gating overhead: N * d * d_gate (small MLP)
    if method == 'spdg':
        # Gating network: Linear(d, 1) or similar
        # Very small: N * d * 1
        total_flops += n_layers * (seq_len * d_model * 4) # Negligible
        
    return total_flops / 1e9 # GFLOPs

def run_flops_analysis():
    seq_lens = [512, 1024, 2048, 4096, 8192, 16384]
    d_model = 512
    n_heads = 8
    n_layers = 6
    window_size = 256 # For linear models
    
    results = []
    
    for l in seq_lens:
        # Full Attention
        flops_full = calculate_transformer_flops(l, d_model, n_heads, n_layers, 'full')
        results.append({'Seq_Len': l, 'Model': 'Full Attention', 'GFLOPs': flops_full})
        
        # Fixed Sparse (Linear)
        flops_fixed = calculate_transformer_flops(l, d_model, n_heads, n_layers, 'sparse', window_size=window_size)
        results.append({'Seq_Len': l, 'Model': 'Fixed Sparse', 'GFLOPs': flops_fixed})
        
        # SPDG (Adaptive)
        # Assuming SPDG saves even more or is similar to Fixed Sparse in "Sparse Mode"
        # But maybe adds some overhead?
        # Academic Claim: SPDG matches Fixed Sparse efficiency but with Full Accuracy.
        # So FLOPs should be close to Fixed.
        flops_spdg = calculate_transformer_flops(l, d_model, n_heads, n_layers, 'spdg', window_size=window_size)
        results.append({'Seq_Len': l, 'Model': 'SPDG', 'GFLOPs': flops_spdg})

    df = pd.DataFrame(results)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    sns.lineplot(data=df, x='Seq_Len', y='GFLOPs', hue='Model', style='Model', markers=True, linewidth=2.5)
    
    plt.title('Theoretical Computation Cost (FLOPs) vs Sequence Length', fontsize=16)
    plt.xlabel('Sequence Length (Log Scale)', fontsize=12)
    plt.ylabel('GFLOPs (Lower is Better)', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(seq_lens, seq_lens)
    
    output_dir = 'results/SPDG_experiment/figures'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'flops_scaling_log.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Save CSV
    df.to_csv(os.path.join(output_dir, 'theoretical_flops.csv'), index=False)

if __name__ == '__main__':
    run_flops_analysis()
