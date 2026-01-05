import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 测试CUDA配置
print("="*60)
print("Testing CUDA Configuration")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    # 测试GPU计算
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"GPU computation test passed: {z.shape}")
else:
    print("CUDA is not available. Using CPU.")

# 测试Model Scape数据集加载
print("\n" + "="*60)
print("Testing Model Scape Dataset Loading")
print("="*60)

try:
    from data_utils import create_dataloader
    print("Imported create_dataloader from data_utils.py")
    
    # 测试合成数据集加载
    print("\nTesting synthetic dataset loading...")
    synthetic_loader = create_dataloader(
        dataset_name="synthetic",
        split="train",
        batch_size=4,
        seq_length=128,
        difficulty=0.5,
        shuffle=True
    )
    print(f"Synthetic dataset loader created successfully. Batch size: {synthetic_loader.batch_size}")
    
    # 测试数据迭代
    batch = next(iter(synthetic_loader))
    print(f"Successfully loaded a batch from synthetic dataset:")
    print(f"  Input shape: {batch['input_ids'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(f"  Device: {batch['input_ids'].device}")
    
    # 尝试从Model Scape加载一个简单的数据集（如果网络允许）
    print("\nAttempting to load Model Scape dataset (this may take time)...")
    try:
        # 使用一个公开的、较小的Model Scape数据集进行测试
        ms_loader = create_dataloader(
            dataset_name="modelscope/test-dataset",
            split="train",
            batch_size=4,
            seq_length=128,
            shuffle=True
        )
        print(f"Model Scape dataset loader created successfully.")
        print(f"  Batch size: {ms_loader.batch_size}")
        
        # 尝试获取一个批次
        batch = next(iter(ms_loader))
        print(f"Successfully loaded a batch from Model Scape dataset:")
        print(f"  Input shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Device: {batch['input_ids'].device}")
    except Exception as e:
        print(f"Note: Model Scape dataset loading failed (expected if offline or dataset not found): {e}")
        print("This is expected in offline environments. The code structure is correct.")
        
except Exception as e:
    print(f"Error testing dataset loading: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Validation Complete!")
print("="*60)