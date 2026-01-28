import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np

# 尝试导入ModelScope，处理Python 3.8兼容性问题
try:
    from modelscope import MsDataset
    HAS_MODELSCOPE = True
except Exception as e:
    print(f"Warning: Failed to import ModelScope. Some datasets may not be available: {e}")
    print("This is likely due to Python version compatibility (ModelScope may require Python 3.9+).")
    HAS_MODELSCOPE = False

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=512, vocab_size=10000, difficulty=0.5):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.difficulty = difficulty
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 根据难度调整序列的随机性
        if self.difficulty < 0.3:
            # 简单难度：大部分是重复模式
            pattern = torch.randint(0, self.vocab_size, (10,))
            input_ids = pattern.repeat(self.seq_length // 10 + 1)[:self.seq_length]
        elif self.difficulty < 0.7:
            # 中等难度：部分重复，部分随机
            pattern = torch.randint(0, self.vocab_size, (20,))
            random_part = torch.randint(0, self.vocab_size, (self.seq_length // 2,))
            mixed = torch.cat([pattern.repeat(self.seq_length // 20 + 1)[:self.seq_length // 2], random_part])
            input_ids = mixed[torch.randperm(mixed.size(0))][:self.seq_length]
        else:
            # 困难难度：完全随机
            input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        
        # 生成分类标签：0或1
        # 根据序列的随机性程度生成标签
        # 简单序列为0，复杂序列为1
        if self.difficulty < 0.3:
            label = torch.tensor(0)
        elif self.difficulty < 0.7:
            # 中等难度随机生成0或1
            label = torch.randint(0, 2, (1,)).squeeze()
        else:
            label = torch.tensor(1)
        
        # 创建全1的attention_mask，表示所有位置都应该被关注
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

def create_dataloader(dataset_name="synthetic", split="train", batch_size=8, seq_length=512, difficulty=0.5, num_workers=0, shuffle=True, **kwargs):
    """创建数据集加载器"""
    # 从kwargs中提取可能的参数
    max_length = kwargs.get('max_length', seq_length)
    num_samples = kwargs.get('num_samples', 1000)
    task_name = kwargs.get('task_name', None)
    
    # 使用max_length覆盖seq_length（如果提供）
    seq_length = max_length
    
    if dataset_name == "synthetic":
        dataset = SyntheticDataset(num_samples=num_samples, seq_length=seq_length, difficulty=difficulty)
    else:
        # 尝试从Hugging Face datasets加载数据集
        try:
            # 使用Hugging Face datasets加载真实数据集
            if dataset_name == 'glue' and task_name:
                # 对于glue数据集，需要指定task_name
                dataset = load_dataset(dataset_name, task_name, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            # 对真实数据集进行预处理
            def preprocess_function(examples):
                # 简单的tokenization和截断处理
                input_ids = examples.get('input_ids', examples.get('text_ids', examples.get('ids', [])))
                labels = examples.get('labels', examples.get('label', 0))
                
                # 确保input_ids是列表格式
                if not isinstance(input_ids, list):
                    input_ids = [input_ids]
                
                # 截断或填充到指定长度
                if len(input_ids) > seq_length:
                    input_ids = input_ids[:seq_length]
                else:
                    # 填充到seq_length（使用0作为padding token）
                    input_ids = input_ids + [0] * (seq_length - len(input_ids))
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': [1] * min(len(examples.get('input_ids', [])), seq_length) + [0] * max(0, seq_length - len(examples.get('input_ids', []))),
                    'labels': labels
                }
            
            dataset = dataset.map(preprocess_function)
        except Exception as e:
            # 如果Hugging Face加载失败，尝试从Model Scape加载
            print(f"Failed to load from Hugging Face: {e}")
            
            if HAS_MODELSCOPE:
                print("Trying ModelScope...")
                
                # ModelScope dataset loading
                # For GLUE, ModelScope uses 'glue' as dataset_name and task_name as subset_name
                if dataset_name == 'glue' and task_name:
                    dataset = MsDataset.load(
                        dataset_name,
                        subset_name=task_name,
                        split=split,
                        **kwargs
                    )
                else:
                    dataset = MsDataset.load(
                        dataset_name,
                        split=split,
                        **kwargs
                    )
                
                # 对Model Scape数据集进行预处理
                def preprocess_ms_function(examples):
                    # 根据Model Scape数据集的具体格式进行处理
                    input_ids = examples.get('input_ids', examples.get('text_ids', examples.get('ids', [])))
                    labels = examples.get('labels', examples.get('label', 0))
                    
                    # 确保input_ids是列表格式
                    if not isinstance(input_ids, list):
                        input_ids = [input_ids]
                    
                    # 截断或填充到指定长度
                    if len(input_ids) > seq_length:
                        input_ids = input_ids[:seq_length]
                    else:
                        # 填充到seq_length（使用0作为padding token）
                        input_ids = input_ids + [0] * (seq_length - len(input_ids))
                    
                    return {
                        'input_ids': input_ids,
                        'attention_mask': [1] * min(len(examples.get('input_ids', [])), seq_length) + [0] * max(0, seq_length - len(examples.get('input_ids', []))),
                        'labels': labels
                    }
                
                # Model Scape数据集可能需要转换为Hugging Face数据集格式
                if hasattr(dataset, 'to_hf_dataset'):
                    dataset = dataset.to_hf_dataset()
                    dataset = dataset.map(preprocess_ms_function)
                else:
                    # 如果是其他格式，转换为列表并处理
                    processed_data = []
                    for data in dataset:
                        processed = preprocess_ms_function(data)
                        processed_data.append(processed)
                    
                    # 创建一个简单的Dataset类来包装处理后的数据
                    class ModelScopeDataset(Dataset):
                        def __init__(self, data):
                            self.data = data
                        
                        def __len__(self):
                            return len(self.data)
                        
                        def __getitem__(self, idx):
                            return self.data[idx]
                    
                    dataset = ModelScopeDataset(processed_data)
            else:
                # 如果ModelScope不可用，抛出原始错误
                print("ModelScope is not available. Cannot load dataset.")
                raise e
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=None
    )
    
    return dataloader

def generate_difficulty_mapping(dataset_name="synthetic"):
    """生成数据集难度映射"""
    if dataset_name == "synthetic":
        # 对于合成数据，我们直接返回难度映射
        return {
            'easy': 0.2,
            'medium': 0.5,
            'hard': 0.8
        }
    else:
        # 对于真实数据集，我们可以根据某些指标计算难度
        # 这里简化处理，返回固定映射
        return {
            'easy': 0.2,
            'medium': 0.5,
            'hard': 0.8
        }