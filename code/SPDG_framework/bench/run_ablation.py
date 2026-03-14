import torch
import os
import json
import time
from train import Trainer, ExperimentConfig
from spdg_transformer import SPDGTransformer
from spdg_components import SPDGTransformerLayer
from data_utils import create_dataloader

class AblationRunner:
    def __init__(self, config):
        self.config = config
        self.output_dir = os.path.join(config.output_dir, 'ablation')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def build_model(self, case_id):
        vocab_size = self.config.dataset_configs['synthetic']['vocab_size']
        d_model = 256
        n_heads = 4
        n_layers = 2
        seq_len = 128
        
        # Base Model Shell
        model = SPDGTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            seq_len=seq_len,
            n_classes=2
        )
        
        # Override layers with ablation configs
        layers = []
        for _ in range(n_layers):
            if case_id == 'M0': # Full Attention
                layer = SPDGTransformerLayer(d_model, n_heads, seq_len, gating_mode='fixed_full', sparsity=0.0)
            elif case_id == 'M1': # Fixed Sparse (Ablated A)
                layer = SPDGTransformerLayer(d_model, n_heads, seq_len, gating_mode='fixed_sparse', sparsity=0.1)
            elif case_id == 'M2': # Uncalibrated SPDG (Ablated B)
                layer = SPDGTransformerLayer(d_model, n_heads, seq_len, calibrate=False)
            elif case_id == 'M3': # Random Sparse SPDG (Ablated C)
                layer = SPDGTransformerLayer(d_model, n_heads, seq_len, pattern='random', sparsity=0.1)
            else: # M4: Full SPDG
                layer = SPDGTransformerLayer(d_model, n_heads, seq_len, sparsity=0.1)
            layers.append(layer)
        
        model.layers = torch.nn.ModuleList(layers)
        return model.to(self.config.device)

    def run(self):
        cases = {
            'M0': 'Full Attention',
            'M1': 'Fixed Sparse',
            'M2': 'Uncalibrated SPDG',
            'M3': 'Random Prior SPDG',
            'M4': 'Full SPDG'
        }
        
        all_results = {}
        
        # Fixed: Explicitly set max_length=128 to match model
        train_loader = create_dataloader('synthetic', split='train', batch_size=16, num_samples=500, max_length=128)
        val_loader = create_dataloader('synthetic', split='validation', batch_size=16, num_samples=200, max_length=128)
        
        for cid, name in cases.items():
            print(f"\n>>> Running Ablation Case {cid}: {name}")
            model = self.build_model(cid)
            
            # Use smaller epochs for fast ablation
            self.config.training_configs['num_epochs'] = 5
            trainer = Trainer(model, self.config, model_name=f"ablation_{cid}")
            
            start_time = time.time()
            metrics = trainer.train(train_loader, val_loader)
            end_time = time.time()
            
            # Store final metrics
            all_results[cid] = {
                'name': name,
                'final_val_acc': metrics['val_acc'][-1],
                'final_val_loss': metrics['val_loss'][-1],
                'time_per_epoch': (end_time - start_time) / 5
            }
            
        with open(os.path.join(self.output_dir, 'ablation_results.json'), 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\nAblation results saved to {self.output_dir}")

if __name__ == "__main__":
    config = ExperimentConfig()
    runner = AblationRunner(config)
    runner.run()
