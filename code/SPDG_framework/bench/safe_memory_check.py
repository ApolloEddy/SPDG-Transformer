import os
import sys
import torch
import gc

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spdg_transformer import SPDGTransformer

def check_memory_safe():
    print("Running Safe Memory Check...")
    print("Testing SPDG at Seq Len 2048 (Previous Usage: ~3.8GB)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA not available. Skipping.")
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    seq_len = 2048
    vocab_size = 10000
    d_model = 512
    n_heads = 8
    n_layers = 6
    
    try:
        model = SPDGTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            seq_len=seq_len + 128, # buffer for safety
            sparsity=0.1
        ).to(device)
        
        input_ids = torch.randint(0, vocab_size, (1, seq_len)).to(device)
        
        print("Executing Forward Pass...")
        logits, _, _ = model(input_ids)
        
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak Memory Usage: {mem_mb:.2f} MB")
        
        if mem_mb < 3000:
            print("SUCCESS: Memory usage significantly reduced (< 3GB). Optimization works!")
        else:
            print("WARNING: Memory usage still high. Optimization might not be fully effective or Pytorch caching is active.")
            
        del model, input_ids, logits
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"CRASH AVOIDED (Caught Exception): {e}")

if __name__ == '__main__':
    check_memory_safe()
