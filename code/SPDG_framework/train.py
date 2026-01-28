import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime

from spdg_transformer import SPDGTransformer
from full_attention_transformer import FullAttentionTransformer
from fixed_sparse_transformer import FixedSparseTransformer
from data_utils import create_dataloader, generate_difficulty_mapping


class ExperimentConfig:
    def __init__(self):
        self.model_configs = {
            'spdg': {
                'd_model': 512,
                'n_heads': 8,
                'n_layers': 6,
                'seq_len': 512,
                'sparsity': 0.1,
                'pattern': 'local',
                'dropout': 0.1
            },
            'full': {
                'd_model': 512,
                'n_heads': 8,
                'n_layers': 6,
                'seq_len': 512,
                'dropout': 0.1
            },
            'fixed': {
                'd_model': 512,
                'n_heads': 8,
                'n_layers': 6,
                'seq_len': 512,
                'sparsity': 0.1,
                'pattern': 'local',
                'dropout': 0.1
            }
        }
        
        self.training_configs = {
            'batch_size': 16,
            'learning_rate': 5e-5,
            'num_epochs': 10,
            'warmup_steps': 100,
            'weight_decay': 0.01,
            'gradient_clip': 1.0
        }
        
        self.dataset_configs = {
            'synthetic': {
                'num_samples': 1000,
                'seq_length': 512,
                'vocab_size': 10000,
                'difficulty': 0.5,
                'n_classes': 2
            },
            'glue': {
                'task_name': 'sst2',
                'max_length': 512,
                'n_classes': 2
            },
            'wikitext': {
                'max_length': 512,
                'vocab_size': 50257
            },
            'logical_reversal': {
                'num_samples': 10000,
                'max_length': 128,
                'n_classes': 2
            },
            'bit_gps': {
                'max_length': 256,
                'n_classes': 10
            }
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = 'results/SPDG_experiment'
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')
        self.figures_dir = os.path.join(self.output_dir, 'figures')


class Trainer:
    def __init__(self, model: nn.Module, config: ExperimentConfig, model_name: str):
        self.model = model.to(config.device)
        self.config = config
        self.model_name = model_name
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training_configs['learning_rate'],
            weight_decay=config.training_configs['weight_decay']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training_configs['num_epochs']
        )
        
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.metrics_dir, exist_ok=True)
        os.makedirs(config.figures_dir, exist_ok=True)
        
        self.metrics_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)
            
            self.optimizer.zero_grad()
            
            logits, _, _ = self.model(input_ids, attention_mask)
            
            loss = self.criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training_configs['gradient_clip']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                logits, _, _ = self.model(input_ids, attention_mask)
                
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                predictions = logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        num_epochs = self.config.training_configs['num_epochs']
        
        for epoch in range(1, num_epochs + 1):
            start_time = datetime.now()
            
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
            else:
                val_loss, val_acc = 0.0, 0.0
            
            self.scheduler.step()
            
            epoch_time = (datetime.now() - start_time).total_seconds()
            
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['train_acc'].append(train_acc)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['val_acc'].append(val_acc)
            self.metrics_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.metrics_history['epoch_time'].append(epoch_time)
            
            print(f'\nEpoch {epoch}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'  Epoch Time: {epoch_time:.2f}s')
            
            if (epoch % 5 == 0) or (epoch == num_epochs):
                self.save_checkpoint(epoch)
        
        return self.metrics_history
    
    def save_checkpoint(self, epoch: int):
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'{self.model_name}_epoch_{epoch}.pt'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics_history': self.metrics_history
        }, checkpoint_path)
        
        print(f'Checkpoint saved to {checkpoint_path}')


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def run_experiment(
        self,
        dataset_name: str,
        model_type: str = 'spdg'
    ) -> Dict:
        print(f'\n{"="*60}')
        print(f'Running experiment: {model_type} on {dataset_name}')
        print(f'{"="*60}\n')
        
        dataset_config = self.config.dataset_configs[dataset_name]
        model_config = self.config.model_configs[model_type]
        
        try:
            train_loader = create_dataloader(
                dataset_name,
                split='train',
                batch_size=self.config.training_configs['batch_size'],
                shuffle=True,
                **dataset_config
            )
            
            val_loader = create_dataloader(
                dataset_name,
                split='validation',
                batch_size=self.config.training_configs['batch_size'],
                shuffle=False,
                **dataset_config
            )
        except Exception as e:
            print(f"Failed to create dataloaders for {dataset_name}: {e}")
            print("Skipping this experiment due to data loading error (likely network issue or missing data).")
            raise e
        
        vocab_size = dataset_config.get('vocab_size', 30522)
        n_classes = dataset_config.get('n_classes', None)
        
        if model_type == 'spdg':
            model = SPDGTransformer(
                vocab_size=vocab_size,
                n_classes=n_classes,
                **model_config
            )
        elif model_type == 'full':
            model = FullAttentionTransformer(
                vocab_size=vocab_size,
                n_classes=n_classes,
                **model_config
            )
        elif model_type == 'fixed':
            model = FixedSparseTransformer(
                vocab_size=vocab_size,
                n_classes=n_classes,
                **model_config
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        trainer = Trainer(model, self.config, f'{model_type}_{dataset_name}')
        
        metrics = trainer.train(train_loader, val_loader)
        
        return metrics
    
    def run_all_experiments(self):
        datasets = ['synthetic', 'glue']
        models = ['spdg', 'full', 'fixed']
        
        all_results = {}
        
        for dataset in datasets:
            for model in models:
                try:
                    # Clear memory before starting new experiment
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    results = self.run_experiment(dataset, model)
                    all_results[f'{model}_{dataset}'] = results
                    
                    # Clear memory after experiment
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f'Error running {model} on {dataset}: {str(e)}')
                    import traceback
                    traceback.print_exc()
                    all_results[f'{model}_{dataset}'] = None
        
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: Dict):
        results_path = os.path.join(
            self.config.output_dir,
            'experiment_results.json'
        )
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    
    results = runner.run_all_experiments()
    
    print('\n' + '='*60)
    print('All experiments completed!')
    print('='*60)
