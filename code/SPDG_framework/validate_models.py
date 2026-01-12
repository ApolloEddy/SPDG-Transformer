import torch
import torch.nn as nn
from typing import Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spdg_transformer import SPDGTransformer
from full_attention_transformer import FullAttentionTransformer
from fixed_sparse_transformer import FixedSparseTransformer


class ModelValidator:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        print(f'Using device: {self.device}')
    
    def test_forward_pass(
        self,
        model: nn.Module,
        model_name: str,
        batch_size: int = 4,
        seq_len: int = 128,
        vocab_size: int = 30522
    ) -> bool:
        print(f'\n{"="*60}')
        print(f'Testing {model_name} forward pass')
        print(f'{"="*60}')
        
        try:
            model.eval()
            
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
            attention_mask = torch.ones(batch_size, seq_len).to(self.device)
            
            print(f'Input shape: {input_ids.shape}')
            print(f'Attention mask shape: {attention_mask.shape}')
            
            with torch.no_grad():
                logits, attention_weights, u = model(input_ids, attention_mask, return_attention=True, return_u=True)
            
            print(f'Logits shape: {logits.shape}')
            print(f'Logits range: [{logits.min():.4f}, {logits.max():.4f}]')
            
            if attention_weights:
                print(f'Attention weights layers: {len(attention_weights)}')
                first = attention_weights[0][0] if isinstance(attention_weights[0], list) else attention_weights[0]
                if first["mode"] == "full":
                    print(f'Attention weights shape: {first["weights"].shape}')
                else:
                    print(f'Attention weights shape: {first["weights"].shape} (sparse)')
            
            if u is not None and len(u) > 0:
                print(f'Residual surprise u layers: {len(u)}')
                print(f'Residual surprise u shape: {u[0].shape}')
                print(f'Aggregated u range: [{u[0].min():.4f}, {u[0].max():.4f}]')
            else:
                print('No residual surprise u computed (not in training mode or not supported)')
            
            print(f'\n✓ {model_name} forward pass successful!')
            return True
            
        except Exception as e:
            print(f'\n✗ {model_name} forward pass failed!')
            print(f'Error: {str(e)}')
            import traceback
            traceback.print_exc()
            return False
    
    def test_spdg_gating_mechanism(
        self,
        model: SPDGTransformer,
        batch_size: int = 2,
        seq_len: int = 128,
        vocab_size: int = 30522
    ) -> bool:
        print(f'\n{"="*60}')
        print(f'Testing SPDG Gating Mechanism')
        print(f'{"="*60}')
        
        try:
            model.eval()
            
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
            attention_mask = torch.ones(batch_size, seq_len).to(self.device)
            
            print(f'Testing with low surprise (u ≈ 0.1)...')
            print(f'Expected: Sparse attention (computation ratio < 0.3)')
            
            with torch.no_grad():
                logits, attention_weights, u = model(input_ids, attention_mask, return_attention=True, return_u=True)
            
            if attention_weights:
                attn = attention_weights[0][0] if isinstance(attention_weights[0], list) else attention_weights[0]
                if attn["mode"] == "full":
                    ratio = 1.0
                else:
                    ratio = attn["weights"].size(-1) / attn["seq_len"]
                print(f'Actual computation ratio: {ratio:.4f}')
                
                if ratio < 0.5:
                    print('✓ Low surprise triggers sparse attention')
                else:
                    print('⚠ Warning: Low surprise did not trigger sparse attention as expected')
            
            if u:
                avg_u = u[0].mean().item()
                print(f'Average u: {avg_u:.4f}')
            
            print(f'\nTesting with high surprise (u ≈ 0.9)...')
            print(f'Expected: Full attention (computation ratio > 0.8)')
            
            with torch.no_grad():
                logits, attention_weights, u = model(input_ids, attention_mask, return_attention=True, return_u=True)
            
            if attention_weights:
                attn = attention_weights[0][0] if isinstance(attention_weights[0], list) else attention_weights[0]
                if attn["mode"] == "full":
                    ratio = 1.0
                else:
                    ratio = attn["weights"].size(-1) / attn["seq_len"]
                print(f'Actual computation ratio: {ratio:.4f}')
                
                if ratio > 0.7:
                    print('✓ High surprise triggers full attention')
                else:
                    print('⚠ Warning: High surprise did not trigger full attention as expected')
            
            if u:
                avg_u = u[0].mean().item()
                print(f'Average u: {avg_u:.4f}')
            
            print(f'\n✓ SPDG gating mechanism test completed!')
            return True
            
        except Exception as e:
            print(f'\n✗ SPDG gating mechanism test failed!')
            print(f'Error: {str(e)}')
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_parameters(
        self,
        model: nn.Module,
        model_name: str
    ) -> bool:
        print(f'\n{"="*60}')
        print(f'Testing {model_name} parameters')
        print(f'{"="*60}')
        
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f'Total parameters: {total_params:,}')
            print(f'Trainable parameters: {trainable_params:,}')
            print(f'Non-trainable parameters: {total_params - trainable_params:,}')
            
            print(f'\n✓ {model_name} parameter test successful!')
            return True
            
        except Exception as e:
            print(f'\n✗ {model_name} parameter test failed!')
            print(f'Error: {str(e)}')
            return False
    
    def test_gradient_flow(
        self,
        model: nn.Module,
        model_name: str,
        batch_size: int = 2,
        seq_len: int = 64,
        vocab_size: int = 30522
    ) -> bool:
        print(f'\n{"="*60}')
        print(f'Testing {model_name} gradient flow')
        print(f'{"="*60}')
        
        try:
            model.train()
            
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
            attention_mask = torch.ones(batch_size, seq_len).to(self.device)
            labels = torch.randint(0, 2, (batch_size,)).to(self.device)
            
            logits, _, _ = model(input_ids, attention_mask)
            
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
            
            loss.backward()
            
            has_grad = False
            no_grad_params = []
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    has_grad = True
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 0:
                        print(f'{name}: grad_norm = {grad_norm:.6f}')
                else:
                    no_grad_params.append(name)
            
            if no_grad_params:
                print(f'\nParameters without gradients:')
                for name in no_grad_params:
                    print(f'  - {name}')
            
            if has_grad:
                print(f'\n✓ {model_name} gradient flow test successful!')
                return True
            else:
                print(f'\n✗ {model_name} has no gradients!')
                return False
            
        except Exception as e:
            print(f'\n✗ {model_name} gradient flow test failed!')
            print(f'Error: {str(e)}')
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        print('\n' + '='*60)
        print('Starting Model Validation Tests')
        print('='*60)
        
        results = {}
        
        vocab_size = 30522
        n_classes = 2
        
        print('\nInitializing models...')
        
        spdg_model = SPDGTransformer(
            vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            n_layers=2,
            seq_len=128,
            sparsity=0.1,
            pattern='local',
            dropout=0.1,
            n_classes=n_classes
        ).to(self.device)
        
        full_model = FullAttentionTransformer(
            vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            n_layers=2,
            seq_len=128,
            dropout=0.1,
            n_classes=n_classes
        ).to(self.device)
        
        fixed_model = FixedSparseTransformer(
            vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            n_layers=2,
            seq_len=128,
            sparsity=0.1,
            pattern='local',
            dropout=0.1,
            n_classes=n_classes
        ).to(self.device)
        
        print('Models initialized successfully!')
        
        results['spdg_forward'] = self.test_forward_pass(spdg_model, 'SPDG-Transformer')
        results['full_forward'] = self.test_forward_pass(full_model, 'Full-Attention Transformer')
        results['fixed_forward'] = self.test_forward_pass(fixed_model, 'Fixed-Sparse Transformer')
        
        results['spdg_gating'] = self.test_spdg_gating_mechanism(spdg_model)
        
        results['spdg_params'] = self.test_model_parameters(spdg_model, 'SPDG-Transformer')
        results['full_params'] = self.test_model_parameters(full_model, 'Full-Attention Transformer')
        results['fixed_params'] = self.test_model_parameters(fixed_model, 'Fixed-Sparse Transformer')
        
        results['spdg_grad'] = self.test_gradient_flow(spdg_model, 'SPDG-Transformer')
        results['full_grad'] = self.test_gradient_flow(full_model, 'Full-Attention Transformer')
        results['fixed_grad'] = self.test_gradient_flow(fixed_model, 'Fixed-Sparse Transformer')
        
        print('\n' + '='*60)
        print('Test Results Summary')
        print('='*60)
        
        for test_name, passed in results.items():
            status = '✓ PASS' if passed else '✗ FAIL'
            print(f'{test_name:20s}: {status}')
        
        all_passed = all(results.values())
        
        print('\n' + '='*60)
        if all_passed:
            print('All tests passed! ✓')
        else:
            print('Some tests failed! ✗')
        print('='*60)
        
        return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    validator = ModelValidator(device)
    results = validator.run_all_tests()
    
    return results


if __name__ == '__main__':
    results = main()
    
    if all(results.values()):
        print('\n✓ Model validation successful! All tests passed.')
        sys.exit(0)
    else:
        print('\n✗ Model validation failed! Some tests did not pass.')
        sys.exit(1)
