import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import json

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


class ExperimentVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        self.color_palette = {
            'spdg': '#2E86AB',
            'full': '#A23B72',
            'fixed': '#F18F01'
        }
        
        self.marker_styles = {
            'spdg': 'o',
            'full': 's',
            'fixed': '^'
        }
    
    def plot_difficulty_vs_computation(
        self,
        data: pd.DataFrame,
        save_name: str = 'difficulty_vs_computation.png'
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model in data['model_name'].unique():
            model_data = data[data['model_name'] == model].sort_values('aggregated_u')
            
            ax.plot(
                model_data['aggregated_u'],
                model_data['computation_ratio'],
                marker=self.marker_styles.get(model, 'o'),
                label=model.upper(),
                color=self.color_palette.get(model, None),
                linewidth=2,
                markersize=8
            )
        
        ax.set_xlabel('Aggregated Surprise (u)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Computation Ratio', fontsize=14, fontweight='bold')
        ax.set_title('Dynamic Efficiency: Computation vs Input Difficulty', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f'Figure saved to {save_path}')
        return save_path
    
    def plot_accuracy_comparison(
        self,
        data: pd.DataFrame,
        save_name: str = 'accuracy_comparison.png'
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = data['model_name'].unique()
        x_pos = np.arange(len(model_names))
        
        accuracies = [data[data['model_name'] == model]['accuracy'].mean() for model in model_names]
        
        bars = ax.bar(
            x_pos,
            accuracies,
            color=[self.color_palette.get(model, '#666666') for model in model_names],
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5
        )
        
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in model_names], fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.02,
                f'{acc:.4f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f'Figure saved to {save_path}')
        return save_path
    
    def plot_inference_time_comparison(
        self,
        data: pd.DataFrame,
        save_name: str = 'inference_time_comparison.png'
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        model_names = data['model_name'].unique()
        x_pos = np.arange(len(model_names))
        
        inference_times = [data[data['model_name'] == model]['mean_inference_time'].mean() 
                          for model in model_names]
        throughputs = [data[data['model_name'] == model]['throughput'].mean() 
                      for model in model_names]
        
        bars1 = ax1.bar(
            x_pos,
            inference_times,
            color=[self.color_palette.get(model, '#666666') for model in model_names],
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5
        )
        
        ax1.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Inference Time (s)', fontsize=14, fontweight='bold')
        ax1.set_title('Inference Time Comparison', fontsize=16, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([m.upper() for m in model_names], fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, time in zip(bars1, inference_times):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height + max(inference_times) * 0.02,
                f'{time:.4f}s',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        bars2 = ax2.bar(
            x_pos,
            throughputs,
            color=[self.color_palette.get(model, '#666666') for model in model_names],
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5
        )
        
        ax2.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Throughput (samples/s)', fontsize=14, fontweight='bold')
        ax2.set_title('Throughput Comparison', fontsize=16, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([m.upper() for m in model_names], fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, tp in zip(bars2, throughputs):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.,
                height + max(throughputs) * 0.02,
                f'{tp:.2f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f'Figure saved to {save_path}')
        return save_path
    
    def plot_training_curves(
        self,
        data: pd.DataFrame,
        save_name: str = 'training_curves.png'
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for model in data['model_name'].unique():
            model_data = data[data['model_name'] == model].sort_values('epoch')
            
            axes[0, 0].plot(
                model_data['epoch'],
                model_data['train_loss'],
                marker=self.marker_styles.get(model, 'o'),
                label=model.upper(),
                color=self.color_palette.get(model, None),
                linewidth=2,
                markersize=6
            )
            
            axes[0, 1].plot(
                model_data['epoch'],
                model_data['train_acc'],
                marker=self.marker_styles.get(model, 'o'),
                label=model.upper(),
                color=self.color_palette.get(model, None),
                linewidth=2,
                markersize=6
            )
            
            axes[1, 0].plot(
                model_data['epoch'],
                model_data['val_loss'],
                marker=self.marker_styles.get(model, 'o'),
                label=model.upper(),
                color=self.color_palette.get(model, None),
                linewidth=2,
                markersize=6
            )
            
            axes[1, 1].plot(
                model_data['epoch'],
                model_data['val_acc'],
                marker=self.marker_styles.get(model, 'o'),
                label=model.upper(),
                color=self.color_palette.get(model, None),
                linewidth=2,
                markersize=6
            )
        
        axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f'Figure saved to {save_path}')
        return save_path
    
    def plot_flops_comparison(
        self,
        data: pd.DataFrame,
        save_name: str = 'flops_comparison.png'
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = data['model_name'].unique()
        x_pos = np.arange(len(model_names))
        
        flops_per_token = [data[data['model_name'] == model]['flops_per_token'].mean() 
                          for model in model_names]
        
        bars = ax.bar(
            x_pos,
            flops_per_token,
            color=[self.color_palette.get(model, '#666666') for model in model_names],
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5
        )
        
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('FLOPs per Token', fontsize=14, fontweight='bold')
        ax.set_title('Computational Efficiency (FLOPs)', fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in model_names], fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, flops in zip(bars, flops_per_token):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + max(flops_per_token) * 0.02,
                f'{flops:.2e}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f'Figure saved to {save_path}')
        return save_path
    
    def plot_efficiency_accuracy_tradeoff(
        self,
        data: pd.DataFrame,
        save_name: str = 'efficiency_accuracy_tradeoff.png'
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model in data['model_name'].unique():
            model_data = data[data['model_name'] == model]
            
            avg_accuracy = model_data['accuracy'].mean()
            avg_computation = model_data['mean_computation_ratio'].mean()
            
            ax.scatter(
                avg_computation,
                avg_accuracy,
                s=500,
                label=model.upper(),
                color=self.color_palette.get(model, '#666666'),
                marker=self.marker_styles.get(model, 'o'),
                edgecolor='black',
                linewidth=2,
                alpha=0.8
            )
            
            ax.annotate(
                model.upper(),
                (avg_computation, avg_accuracy),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold'
            )
        
        ax.set_xlabel('Average Computation Ratio', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Efficiency-Accuracy Tradeoff', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1.1])
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f'Figure saved to {save_path}')
        return save_path
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        save_name: str = 'attention_heatmap.png',
        title: str = 'Attention Pattern'
    ):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
        
        ax.set_xlabel('Key Position', fontsize=14, fontweight='bold')
        ax.set_ylabel('Query Position', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f'Figure saved to {save_path}')
        return save_path
    
    def plot_sparsity_distribution(
        self,
        data: pd.DataFrame,
        save_name: str = 'sparsity_distribution.png'
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model in data['model_name'].unique():
            model_data = data[data['model_name'] == model]
            
            ax.hist(
                model_data['mean_sparsity'],
                bins=20,
                alpha=0.6,
                label=model.upper(),
                color=self.color_palette.get(model, '#666666'),
                edgecolor='black',
                linewidth=1
            )
        
        ax.set_xlabel('Sparsity Ratio', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title('Sparsity Distribution Across Models', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f'Figure saved to {save_path}')
        return save_path
    
    def generate_all_figures(self, metrics_dir: str) -> List[str]:
        figure_paths = []
        
        training_data = None
        inference_data = None
        difficulty_data = None
        
        training_path = os.path.join(metrics_dir, 'training_metrics.csv')
        if os.path.exists(training_path):
            training_data = pd.read_csv(training_path)
        
        inference_path = os.path.join(metrics_dir, 'inference_metrics.csv')
        if os.path.exists(inference_path):
            inference_data = pd.read_csv(inference_path)
        
        difficulty_path = os.path.join(metrics_dir, 'difficulty_experiment_metrics.csv')
        if os.path.exists(difficulty_path):
            difficulty_data = pd.read_csv(difficulty_path)
        
        if training_data is not None:
            path = self.plot_training_curves(training_data)
            figure_paths.append(path)
        
        if inference_data is not None:
            path = self.plot_accuracy_comparison(inference_data)
            figure_paths.append(path)
            
            path = self.plot_inference_time_comparison(inference_data)
            figure_paths.append(path)
            
            path = self.plot_flops_comparison(inference_data)
            figure_paths.append(path)
            
            path = self.plot_efficiency_accuracy_tradeoff(inference_data)
            figure_paths.append(path)
            
            path = self.plot_sparsity_distribution(inference_data)
            figure_paths.append(path)
        
        if difficulty_data is not None:
            path = self.plot_difficulty_vs_computation(difficulty_data)
            figure_paths.append(path)
        
        return figure_paths
    
    def create_summary_figure(
        self,
        metrics_dir: str,
        save_name: str = 'summary_figure.png'
    ):
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        training_data = None
        inference_data = None
        difficulty_data = None
        
        training_path = os.path.join(metrics_dir, 'training_metrics.csv')
        if os.path.exists(training_path):
            training_data = pd.read_csv(training_path)
        
        inference_path = os.path.join(metrics_dir, 'inference_metrics.csv')
        if os.path.exists(inference_path):
            inference_data = pd.read_csv(inference_path)
        
        difficulty_path = os.path.join(metrics_dir, 'difficulty_experiment_metrics.csv')
        if os.path.exists(difficulty_path):
            difficulty_data = pd.read_csv(difficulty_path)
        
        ax1 = fig.add_subplot(gs[0, 0])
        if inference_data is not None:
            model_names = inference_data['model_name'].unique()
            x_pos = np.arange(len(model_names))
            accuracies = [inference_data[inference_data['model_name'] == model]['accuracy'].mean() 
                         for model in model_names]
            colors = [self.color_palette.get(model, '#666666') for model in model_names]
            ax1.bar(x_pos, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([m.upper() for m in model_names], fontsize=10)
            ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax1.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
        
        ax2 = fig.add_subplot(gs[0, 1])
        if inference_data is not None:
            model_names = inference_data['model_name'].unique()
            x_pos = np.arange(len(model_names))
            times = [inference_data[inference_data['model_name'] == model]['mean_inference_time'].mean() 
                    for model in model_names]
            colors = [self.color_palette.get(model, '#666666') for model in model_names]
            ax2.bar(x_pos, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([m.upper() for m in model_names], fontsize=10)
            ax2.set_ylabel('Time (s)', fontsize=11, fontweight='bold')
            ax2.set_title('Inference Time', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        ax3 = fig.add_subplot(gs[0, 2])
        if inference_data is not None:
            model_names = inference_data['model_name'].unique()
            x_pos = np.arange(len(model_names))
            flops = [inference_data[inference_data['model_name'] == model]['flops_per_token'].mean() 
                    for model in model_names]
            colors = [self.color_palette.get(model, '#666666') for model in model_names]
            ax3.bar(x_pos, flops, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([m.upper() for m in model_names], fontsize=10)
            ax3.set_ylabel('FLOPs/Token', fontsize=11, fontweight='bold')
            ax3.set_title('Computational Cost', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
        
        ax4 = fig.add_subplot(gs[1, :])
        if training_data is not None:
            for model in training_data['model_name'].unique():
                model_data = training_data[training_data['model_name'] == model].sort_values('epoch')
                ax4.plot(model_data['epoch'], model_data['val_acc'], 
                        marker=self.marker_styles.get(model, 'o'),
                        label=model.upper(), 
                        color=self.color_palette.get(model, None),
                        linewidth=2, markersize=6)
            ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold')
            ax4.set_title('Training Progress', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=10, loc='best')
            ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[2, :])
        if difficulty_data is not None:
            for model in difficulty_data['model_name'].unique():
                model_data = difficulty_data[difficulty_data['model_name'] == model].sort_values('aggregated_u')
                ax5.plot(model_data['aggregated_u'], model_data['computation_ratio'],
                        marker=self.marker_styles.get(model, 'o'),
                        label=model.upper(),
                        color=self.color_palette.get(model, None),
                        linewidth=2, markersize=6)
            ax5.set_xlabel('Aggregated Surprise (u)', fontsize=11, fontweight='bold')
            ax5.set_ylabel('Computation Ratio', fontsize=11, fontweight='bold')
            ax5.set_title('Dynamic Efficiency', fontsize=12, fontweight='bold')
            ax5.legend(fontsize=10, loc='best')
            ax5.grid(True, alpha=0.3)
        
        plt.suptitle('SPDG-Transformer Experiment Summary', fontsize=16, fontweight='bold', y=0.995)
        
        save_path = os.path.join(self.figures_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f'Summary figure saved to {save_path}')
        return save_path
