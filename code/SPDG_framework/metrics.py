import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os
from collections import defaultdict

from spdg_transformer import SPDGTransformer
from full_attention_transformer import FullAttentionTransformer
from fixed_sparse_transformer import FixedSparseTransformer


class MetricsCalculator:
    def __init__(self, device: torch.device):
        self.device = device
    
    def count_flops(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        model.eval()
        
        batch_size, seq_len = input_ids.shape
        
        if hasattr(model, 'd_model'):
            d_model = model.d_model
            n_heads = model.n_heads
            n_layers = model.n_layers
        else:
            d_model = 512
            n_heads = 8
            n_layers = 6
        
        head_dim = d_model // n_heads
        
        qk_flops = batch_size * n_heads * seq_len * seq_len * head_dim
        v_flops = batch_size * n_heads * seq_len * seq_len * head_dim
        attention_flops = qk_flops + v_flops
        
        ffn_flops = batch_size * seq_len * d_model * (4 * d_model) * 2
        
        total_flops = n_layers * (attention_flops + ffn_flops)
        
        return {
            'qk_flops': qk_flops,
            'attention_flops': attention_flops,
            'ffn_flops': ffn_flops,
            'total_flops': total_flops,
            'flops_per_token': total_flops / (batch_size * seq_len)
        }
    
    def measure_inference_time(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_batches: int = 10,
        warmup: int = 3
    ) -> Dict[str, float]:
        model.eval()
        
        times = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches + warmup:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if i >= warmup:
                    start_time = time.perf_counter()
                    _ = model(input_ids, attention_mask)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                else:
                    _ = model(input_ids, attention_mask)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': len(times) / np.sum(times)
        }
    
    def calculate_accuracy(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        model.eval()
        
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Calculating accuracy'):
                input_ids = batch['input_ids'].to(self.device)
                
                # 检查是否有attention_mask，如果没有则创建全1掩码
                if 'attention_mask' in batch:
                    attention_mask = batch['attention_mask'].to(self.device)
                else:
                    # 创建默认的全1注意力掩码
                    attention_mask = torch.ones_like(input_ids).to(self.device)
                
                labels = batch['labels'].to(self.device)
                
                logits, _, _ = model(input_ids, attention_mask)
                
                predictions = logits.argmax(dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def calculate_sparsity_ratio(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_batches: int = 10
    ) -> Dict[str, float]:
        model.eval()
        
        sparsity_ratios = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                _, attention_weights, _ = model(input_ids, attention_mask, return_attention=True)
                
                if attention_weights:
                    for attn in attention_weights:
                        batch_size, n_heads, seq_len, _ = attn.shape
                        zero_ratio = (attn == 0).float().mean().item()
                        sparsity_ratios.append(zero_ratio)
        
        return {
            'mean_sparsity': np.mean(sparsity_ratios),
            'std_sparsity': np.std(sparsity_ratios),
            'min_sparsity': np.min(sparsity_ratios),
            'max_sparsity': np.max(sparsity_ratios)
        }
    
    def calculate_qk_computation_ratio(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_batches: int = 10
    ) -> Dict[str, float]:
        model.eval()
        
        computation_ratios = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                _, attention_weights, _ = model(input_ids, attention_mask, return_attention=True)
                
                if attention_weights:
                    for attn in attention_weights:
                        batch_size, n_heads, seq_len, _ = attn.shape
                        non_zero_ratio = (attn > 0).float().mean().item()
                        computation_ratios.append(non_zero_ratio)
        
        return {
            'mean_computation_ratio': np.mean(computation_ratios),
            'std_computation_ratio': np.std(computation_ratios),
            'min_computation_ratio': np.min(computation_ratios),
            'max_computation_ratio': np.max(computation_ratios)
        }


class ExperimentLogger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_dir = os.path.join(output_dir, 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        self.experiment_logs = defaultdict(list)
    
    def log_training_metrics(
        self,
        model_name: str,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        learning_rate: float,
        epoch_time: float
    ):
        log_entry = {
            'model_name': model_name,
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': learning_rate,
            'epoch_time': epoch_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.experiment_logs['training'].append(log_entry)
    
    def log_inference_metrics(
        self,
        model_name: str,
        dataset_name: str,
        accuracy: float,
        inference_time: Dict[str, float],
        flops: Dict[str, float],
        sparsity: Optional[Dict[str, float]] = None,
        computation_ratio: Optional[Dict[str, float]] = None
    ):
        log_entry = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'accuracy': accuracy,
            'mean_inference_time': inference_time.get('mean_time', 0.0),
            'throughput': inference_time.get('throughput', 0.0),
            'total_flops': flops.get('total_flops', 0.0),
            'flops_per_token': flops.get('flops_per_token', 0.0),
            'mean_sparsity': sparsity.get('mean_sparsity', 0.0) if sparsity else 0.0,
            'mean_computation_ratio': computation_ratio.get('mean_computation_ratio', 1.0) if computation_ratio else 1.0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.experiment_logs['inference'].append(log_entry)
    
    def log_difficulty_experiment(
        self,
        model_name: str,
        difficulty_level: int,
        aggregated_u: float,
        computation_ratio: float,
        accuracy: float,
        inference_time: float
    ):
        log_entry = {
            'model_name': model_name,
            'difficulty_level': difficulty_level,
            'aggregated_u': aggregated_u,
            'computation_ratio': computation_ratio,
            'accuracy': accuracy,
            'inference_time': inference_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.experiment_logs['difficulty_experiment'].append(log_entry)
    
    def save_logs(self):
        for log_type, logs in self.experiment_logs.items():
            if logs:
                df = pd.DataFrame(logs)
                output_path = os.path.join(self.metrics_dir, f'{log_type}_metrics.csv')
                df.to_csv(output_path, index=False)
                print(f'{log_type.capitalize()} metrics saved to {output_path}')
    
    def get_summary_statistics(self) -> Dict:
        summary = {}
        
        if 'training' in self.experiment_logs and self.experiment_logs['training']:
            train_df = pd.DataFrame(self.experiment_logs['training'])
            summary['training'] = {
                'num_epochs': train_df['epoch'].max(),
                'best_train_acc': train_df['train_acc'].max(),
                'best_val_acc': train_df['val_acc'].max(),
                'total_training_time': train_df['epoch_time'].sum()
            }
        
        if 'inference' in self.experiment_logs and self.experiment_logs['inference']:
            infer_df = pd.DataFrame(self.experiment_logs['inference'])
            summary['inference'] = {
                'num_models': infer_df['model_name'].nunique(),
                'avg_accuracy': infer_df['accuracy'].mean(),
                'avg_inference_time': infer_df['mean_inference_time'].mean(),
                'avg_throughput': infer_df['throughput'].mean()
            }
        
        return summary


class MetricsTracker:
    def __init__(self, output_dir: str, device: torch.device):
        self.calculator = MetricsCalculator(device)
        self.logger = ExperimentLogger(output_dir)
        self.device = device
    
    def evaluate_model(
        self,
        model: nn.Module,
        model_name: str,
        dataloader: DataLoader,
        dataset_name: str,
        compute_flops: bool = True,
        compute_sparsity: bool = True,
        compute_computation_ratio: bool = True
    ) -> Dict:
        print(f'\nEvaluating {model_name} on {dataset_name}...')
        
        accuracy_metrics = self.calculator.calculate_accuracy(model, dataloader)
        
        inference_time_metrics = self.calculator.measure_inference_time(model, dataloader)
        
        flops_metrics = {}
        if compute_flops:
            sample_batch = next(iter(dataloader))
            input_ids = sample_batch['input_ids'].to(self.device)
            attention_mask = sample_batch['attention_mask'].to(self.device)
            flops_metrics = self.calculator.count_flops(model, input_ids, attention_mask)
        
        sparsity_metrics = None
        if compute_sparsity:
            sparsity_metrics = self.calculator.calculate_sparsity_ratio(model, dataloader)
        
        computation_ratio_metrics = None
        if compute_computation_ratio:
            computation_ratio_metrics = self.calculator.calculate_qk_computation_ratio(model, dataloader)
        
        self.logger.log_inference_metrics(
            model_name=model_name,
            dataset_name=dataset_name,
            accuracy=accuracy_metrics['accuracy'],
            inference_time=inference_time_metrics,
            flops=flops_metrics,
            sparsity=sparsity_metrics,
            computation_ratio=computation_ratio_metrics
        )
        
        return {
            'accuracy': accuracy_metrics,
            'inference_time': inference_time_metrics,
            'flops': flops_metrics,
            'sparsity': sparsity_metrics,
            'computation_ratio': computation_ratio_metrics
        }
    
    def run_difficulty_experiment(
        self,
        model: nn.Module,
        model_name: str,
        dataloader: DataLoader,
        difficulty_levels: List[int] = [1, 2, 3, 4, 5]
    ) -> Dict:
        print(f'\nRunning difficulty experiment for {model_name}...')
        
        results = {}
        
        model.eval()
        
        for difficulty in difficulty_levels:
            print(f'  Difficulty level: {difficulty}')
            
            correct = 0
            total = 0
            computation_ratios = []
            inference_times = []
            aggregated_us = []
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f'Difficulty {difficulty}', leave=False):
                    if 'difficulty' in batch:
                        batch_difficulty = batch['difficulty']
                        mask = (batch_difficulty == difficulty)
                        
                        if not mask.any():
                            continue
                        
                        input_ids = batch['input_ids'][mask].to(self.device)
                        attention_mask = batch['attention_mask'][mask].to(self.device)
                        labels = batch['labels'][mask].to(self.device)
                        
                        start_time = time.perf_counter()
                        logits, attention_weights, u_list = model(
                            input_ids,
                            attention_mask,
                            return_attention=True,
                            return_u=True
                        )
                        end_time = time.perf_counter()
                        
                        predictions = logits.argmax(dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
                        
                        inference_times.append(end_time - start_time)
                        
                        if attention_weights:
                            for attn in attention_weights:
                                non_zero_ratio = (attn > 0).float().mean().item()
                                computation_ratios.append(non_zero_ratio)
                        
                        if u_list:
                            for u in u_list:
                                aggregated_u = u.mean().item()
                                aggregated_us.append(aggregated_u)
            
            if total > 0:
                accuracy = correct / total
                avg_computation_ratio = np.mean(computation_ratios) if computation_ratios else 1.0
                avg_inference_time = np.mean(inference_times) if inference_times else 0.0
                avg_aggregated_u = np.mean(aggregated_us) if aggregated_us else 0.5
                
                self.logger.log_difficulty_experiment(
                    model_name=model_name,
                    difficulty_level=difficulty,
                    aggregated_u=avg_aggregated_u,
                    computation_ratio=avg_computation_ratio,
                    accuracy=accuracy,
                    inference_time=avg_inference_time
                )
                
                results[difficulty] = {
                    'accuracy': accuracy,
                    'computation_ratio': avg_computation_ratio,
                    'inference_time': avg_inference_time,
                    'aggregated_u': avg_aggregated_u,
                    'num_samples': total
                }
        
        return results
    
    def save_all_metrics(self):
        self.logger.save_logs()
        
        summary = self.logger.get_summary_statistics()
        
        summary_path = os.path.join(self.logger.metrics_dir, 'summary.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f'Summary statistics saved to {summary_path}')
        
        return summary
