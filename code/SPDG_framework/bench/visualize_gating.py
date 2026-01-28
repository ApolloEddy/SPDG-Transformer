import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import ExperimentConfig, Trainer
from spdg_transformer import SPDGTransformer
from data_utils import SyntheticDataset

def visualize_gating_behavior():
    config = ExperimentConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load SPDG Model
    # Assumes checkpoint exists from --fast run
    checkpoint_path = os.path.join(config.checkpoint_dir, 'spdg_synthetic_epoch_1.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please run training first: python code/SPDG_framework/main.py --fast --device cuda")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Initialize model structure
    # Must match config used in training
    dataset_config = config.dataset_configs['synthetic']
    vocab_size = dataset_config.get('vocab_size', 10000)
    n_classes = dataset_config.get('n_classes', 2)
    
    model = SPDGTransformer(
        vocab_size=vocab_size, 
        n_classes=n_classes,
        **config.model_configs['spdg']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate test samples with varying difficulty
    difficulties = np.linspace(0.0, 1.0, 50) # 50 levels of difficulty
    lambdas_by_difficulty = []
    
    print("Running inference on varying difficulty samples...")
    
    with torch.no_grad():
        for diff in difficulties:
            # Create a dataset with 1 sample of specific difficulty
            # SyntheticDataset difficulty controls randomness
            ds = SyntheticDataset(num_samples=1, seq_length=512, difficulty=diff, vocab_size=vocab_size)
            sample = ds[0]
            input_ids = sample['input_ids'].unsqueeze(0).to(device) # (1, 512)
            
            # Forward pass getting u list
            # We need attention weights or lambda.
            # SPDGTransformer forward returns u_list if return_u=True
            _, _, u_list = model(input_ids, return_u=True)
            
            # u_list contains aggregated_u from each layer.
            # Lambda is computed from u_prev.
            # lambda_l = sigmoid(alpha * u_{l-1} + beta) * lambda_max
            # But we can just look at 'u' itself as proxy for surprise.
            # Or we can hook into detailed values?
            # Let's use the 'u' values returned. 
            # Note: u_list[i] corresponds to output of layer i.
            # Layer i+1 uses u_list[i] for gating.
            
            # Extract mean u across layers (or specific layer)
            # Let's pick middle layer (e.g. index 3 for 6-layer model)
            target_u = u_list[2].item() # Layer 2's output surprise (used by Layer 3)
            lambdas_by_difficulty.append(target_u)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    plt.scatter(difficulties, lambdas_by_difficulty, c=difficulties, cmap='viridis', s=100, edgecolors='black')
    
    # Fit a smooth line (Trend)
    sns.regplot(x=difficulties, y=lambdas_by_difficulty, scatter=False, color="red", line_kws={'linestyle': '--'})
    
    plt.title('SPDG Intelligence Test: Surprise (u) vs Input Difficulty', fontsize=16)
    plt.xlabel('Input Difficulty (0.0=Pattern, 1.0=Random)', fontsize=12)
    plt.ylabel('Model Surprise (u)', fontsize=12)
    plt.text(0.05, 0.95, 'Expectation: Harder input -> Higher Surprise', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.9))
    
    output_dir = config.figures_dir
    save_path = os.path.join(output_dir, 'gating_intelligence.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    visualize_gating_behavior()
