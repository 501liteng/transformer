import os
import torch
import torch.nn as nn
from model import Transformer, TransformerEncoder, MultiHeadAttention, FeedForwardNetwork
from utils import load_data
import matplotlib.pyplot as plt
import time
import numpy as np

class AblationExperiment:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # 设置PyTorch内存分配器
        if torch.cuda.is_available():
            # 启用内存缓存优化
            torch.cuda.empty_cache()
            # 设置分配器配置
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'
            # 设置cudnn基准模式
            torch.backends.cudnn.benchmark = True
        
    def train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        total_loss = 0
        total_batches = len(train_loader)
        
        print(f"\nTraining - {total_batches} batches")
        for i, batch in enumerate(train_loader, 1):
            try:
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                src, tgt = batch
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                optimizer.zero_grad()
                output = model(src, tgt[:, :-1])
                loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                
                loss.backward()
                optimizer.step()
                
                # 立即释放不需要的计算图
                del output
                torch.cuda.empty_cache()
                
                total_loss += loss.item()
                
                # 打印进度
                if i % 10 == 0 or i == total_batches:
                    print(f"\rBatch {i}/{total_batches} - Current loss: {loss.item():.4f}", end="")
                    if torch.cuda.is_available():
                        memory = torch.cuda.memory_allocated() / 1024**3
                        print(f" - GPU Memory: {memory:.2f}GB", end="")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nGPU内存不足，尝试清理缓存...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / total_batches
        print(f"\nEpoch average loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src, tgt = batch
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                output = model(src, tgt[:, :-1])
                loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                total_loss += loss.item()
                
        return total_loss / len(val_loader)

    def run_experiment(self, model_config, train_loader, val_loader, epochs=10):
        model = Transformer(**model_config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            val_loss = self.evaluate(model, val_loader, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }

    def ablation_study(self):
        # 加载数据 - 使用较小的批次大小和序列长度
        train_loader, val_loader, char_to_idx, idx_to_char, vocab_size = load_data(
            batch_size=16,     # 减小批次大小
            seq_length=64      # 减小序列长度
        )

        # 基准配置 - 减小模型规模以节省显存
        base_config = {
            'src_vocab_size': vocab_size,
            'tgt_vocab_size': vocab_size,
            'embedding_dim': 256,    # 减小嵌入维度
            'num_heads': 8,
            'num_layers': 4,        # 减少层数
            'ffn_dim': 1024        # 减小前馈网络维度
        }

        # 1. 多头注意力机制的消融实验
        head_configs = [1, 2, 4, 8]
        for num_heads in head_configs:
            config = base_config.copy()
            config['num_heads'] = num_heads
            self.results[f'heads_{num_heads}'] = self.run_experiment(
                config, train_loader, val_loader
            )

        # 2. 层数的消融实验
        layer_configs = [2, 4, 6, 8]
        for num_layers in layer_configs:
            config = base_config.copy()
            config['num_layers'] = num_layers
            self.results[f'layers_{num_layers}'] = self.run_experiment(
                config, train_loader, val_loader
            )

        # 3. 嵌入维度的消融实验
        dim_configs = [128, 256, 512, 1024]
        for dim in dim_configs:
            config = base_config.copy()
            config['embedding_dim'] = dim
            config['ffn_dim'] = dim * 4
            self.results[f'dim_{dim}'] = self.run_experiment(
                config, train_loader, val_loader
            )

    def plot_results(self):
        plt.figure(figsize=(15, 5))
        
        # 1. 多头注意力结果
        plt.subplot(1, 3, 1)
        for num_heads in [1, 2, 4, 8]:
            plt.plot(self.results[f'heads_{num_heads}']['val_losses'], 
                    label=f'{num_heads} heads')
        plt.title('Number of Attention Heads')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        
        # 2. 层数结果
        plt.subplot(1, 3, 2)
        for num_layers in [2, 4, 6, 8]:
            plt.plot(self.results[f'layers_{num_layers}']['val_losses'], 
                    label=f'{num_layers} layers')
        plt.title('Number of Transformer Layers')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        
        # 3. 嵌入维度结果
        plt.subplot(1, 3, 3)
        for dim in [128, 256, 512, 1024]:
            plt.plot(self.results[f'dim_{dim}']['val_losses'], 
                    label=f'dim {dim}')
        plt.title('Embedding Dimension')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('ablation_results.png')
        plt.close()

def main():
    experiment = AblationExperiment()
    experiment.ablation_study()
    experiment.plot_results()
    
    # 打印最终结果
    print("\nAblation Study Results:")
    print("\n1. Number of Attention Heads:")
    for num_heads in [1, 2, 4, 8]:
        result = experiment.results[f'heads_{num_heads}']
        print(f"{num_heads} heads - Final val loss: {result['final_val_loss']:.4f}")
    
    print("\n2. Number of Transformer Layers:")
    for num_layers in [2, 4, 6, 8]:
        result = experiment.results[f'layers_{num_layers}']
        print(f"{num_layers} layers - Final val loss: {result['final_val_loss']:.4f}")
    
    print("\n3. Embedding Dimension:")
    for dim in [128, 256, 512, 1024]:
        result = experiment.results[f'dim_{dim}']
        print(f"Dimension {dim} - Final val loss: {result['final_val_loss']:.4f}")

if __name__ == "__main__":
    main()