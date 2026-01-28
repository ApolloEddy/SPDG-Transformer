import os
import sys
import argparse
import torch
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import ExperimentRunner, ExperimentConfig
from metrics import MetricsTracker
from visualization import ExperimentVisualizer
from report_generator import generate_experiment_report
from validate_models import ModelValidator


def setup_environment():
    print('='*60)
    print('SPDG-Transformer Experiment Pipeline')
    print('='*60)
    print(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'CUDA Version: {torch.version.cuda}')
    print()


def validate_models(config):
    print('\n' + '='*60)
    print('Step 1: Model Validation')
    print('='*60)
    
    validator = ModelValidator(str(config.device))
    results = validator.run_all_tests()
    
    if not all(results.values()):
        print('\n⚠ Warning: Some model validation tests failed.')
        print('Proceeding with caution...')
    
    return results


def run_experiments(config, skip_training=False):
    print('\n' + '='*60)
    print('Step 2: Running Experiments')
    print('='*60)
    
    runner = ExperimentRunner(config)
    
    if skip_training:
        print('Skipping training phase...')
        return None
    
    results = runner.run_all_experiments()
    
    return results


def evaluate_models(config):
    print('\n' + '='*60)
    print('Step 3: Model Evaluation')
    print('='*60)
    
    from spdg_transformer import SPDGTransformer
    from full_attention_transformer import FullAttentionTransformer
    from fixed_sparse_transformer import FixedSparseTransformer
    from data_utils import create_dataloader
    
    tracker = MetricsTracker(config.output_dir, config.device)
    
    vocab_size = 30522
    n_classes = 2
    
    models = {
        'spdg': SPDGTransformer(
            vocab_size=vocab_size,
            d_model=config.model_configs['spdg']['d_model'],
            n_heads=config.model_configs['spdg']['n_heads'],
            n_layers=config.model_configs['spdg']['n_layers'],
            seq_len=config.model_configs['spdg']['seq_len'],
            sparsity=config.model_configs['spdg']['sparsity'],
            pattern=config.model_configs['spdg']['pattern'],
            dropout=config.model_configs['spdg']['dropout'],
            n_classes=n_classes
        ).to(config.device),
        
        'full': FullAttentionTransformer(
            vocab_size=vocab_size,
            d_model=config.model_configs['full']['d_model'],
            n_heads=config.model_configs['full']['n_heads'],
            n_layers=config.model_configs['full']['n_layers'],
            seq_len=config.model_configs['full']['seq_len'],
            dropout=config.model_configs['full']['dropout'],
            n_classes=n_classes
        ).to(config.device),
        
        'fixed': FixedSparseTransformer(
            vocab_size=vocab_size,
            d_model=config.model_configs['fixed']['d_model'],
            n_heads=config.model_configs['fixed']['n_heads'],
            n_layers=config.model_configs['fixed']['n_layers'],
            seq_len=config.model_configs['fixed']['seq_len'],
            sparsity=config.model_configs['fixed']['sparsity'],
            pattern=config.model_configs['fixed']['pattern'],
            dropout=config.model_configs['fixed']['dropout'],
            n_classes=n_classes
        ).to(config.device)
    }
    
    datasets = ['synthetic']
    
    for dataset_name in datasets:
        print(f'\nEvaluating on {dataset_name}...')
        
        val_loader = create_dataloader(
            dataset_name,
            split='validation',
            batch_size=16,
            shuffle=False,
            difficulty=0.5,
            max_length=512
        )
        
        for model_name, model in models.items():
            print(f'  {model_name.upper()}...')
            tracker.evaluate_model(
                model=model,
                model_name=model_name,
                dataloader=val_loader,
                dataset_name=dataset_name
            )
    
    summary = tracker.save_all_metrics()
    
    return summary


def generate_visualizations(config):
    print('\n' + '='*60)
    print('Step 4: Generating Visualizations')
    print('='*60)
    
    visualizer = ExperimentVisualizer(config.output_dir)
    
    figure_paths = visualizer.generate_all_figures(config.metrics_dir)
    
    summary_path = visualizer.create_summary_figure(config.metrics_dir)
    figure_paths.append(summary_path)
    
    print(f'\nGenerated {len(figure_paths)} figures:')
    for path in figure_paths:
        print(f'  - {path}')
    
    return figure_paths


def generate_report(config):
    print('\n' + '='*60)
    print('Step 5: Generating Report')
    print('='*60)
    
    report_path = generate_experiment_report(
        output_dir=config.output_dir,
        experiment_name='SPDG-Transformer Experiment',
        author='SPDG Research Team'
    )
    
    print(f'\nReport generated: {report_path}')
    
    return report_path


def main():
    parser = argparse.ArgumentParser(description='SPDG-Transformer Experiment Pipeline')
    parser.add_argument('--skip-training', action='store_true', help='Skip training phase')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip evaluation phase')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip visualization phase')
    parser.add_argument('--skip-report', action='store_true', help='Skip report generation')
    parser.add_argument('--validate-only', action='store_true', help='Only run model validation')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--fast', action='store_true', help='Run fast experiment with fewer epochs and samples')
    
    args = parser.parse_args()
    
    setup_environment()
    
    config = ExperimentConfig()
    
    if args.fast:
        print("Running in FAST mode: 1 epoch, 100 samples.")
        config.training_configs['num_epochs'] = 1
        config.training_configs['batch_size'] = 8
        if 'synthetic' in config.dataset_configs:
            config.dataset_configs['synthetic']['num_samples'] = 100
        if 'logical_reversal' in config.dataset_configs:
            config.dataset_configs['logical_reversal']['num_samples'] = 100
    
    if args.device != 'auto':
        config.device = torch.device(args.device)
    
    validation_results = validate_models(config)
    
    if args.validate_only:
        print('\nValidation only mode. Exiting...')
        return
    
    training_results = None
    if not args.skip_training:
        training_results = run_experiments(config, args.skip_training)
    
    evaluation_summary = None
    if not args.skip_evaluation:
        evaluation_summary = evaluate_models(config)
    
    figure_paths = None
    if not args.skip_visualization:
        figure_paths = generate_visualizations(config)
    
    report_path = None
    if not args.skip_report:
        report_path = generate_report(config)
    
    print('\n' + '='*60)
    print('Pipeline Completed!')
    print('='*60)
    print(f'End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    print('Output files:')
    if training_results:
        print(f'  - Training results: {os.path.join(config.output_dir, "experiment_results.json")}')
    if evaluation_summary:
        print(f'  - Metrics: {config.metrics_dir}')
    if figure_paths:
        print(f'  - Figures: {config.figures_dir}')
    if report_path:
        print(f'  - Report: {report_path}')
    
    print('\n✓ All tasks completed successfully!')


if __name__ == '__main__':
    main()
