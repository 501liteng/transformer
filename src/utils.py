import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_shakespeare_data(file_path=None):
    """Load and preprocess the Shakespeare dataset."""
    if file_path is None:
        import os
        # 获取当前文件的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建数据文件的绝对路径
        file_path = os.path.join(os.path.dirname(current_dir), 'data', 'tiny_shakespeare.txt')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Shakespeare数据文件未找到: {file_path}\n"
            "请确保在项目根目录的data文件夹中包含tiny_shakespeare.txt文件。\n"
            "您可以从以下地址下载数据文件：\n"
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )


def create_char_mappings(text):
    """Create character to index and index to character mappings."""
    chars = sorted(list(set(text)))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    vocab_size = len(chars)
    return char_to_idx, idx_to_char, vocab_size


def text_to_sequence(text, char_to_idx):
    """Convert text to sequence of indices."""
    return [char_to_idx[char] for char in text]


class ShakespeareDataset(Dataset):
    def __init__(self, text, char_to_idx, seq_length):
        self.text = text
        self.char_to_idx = char_to_idx
        self.seq_length = seq_length
        self.data = text_to_sequence(text, char_to_idx)
        self.total_seq = len(self.data) - seq_length - 1

    def __len__(self):
        return self.total_seq

    def __getitem__(self, idx):
        # Get input sequence and target sequence
        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        
        return (torch.tensor(input_seq), 
                torch.tensor(target_seq))


def load_data(batch_size=16, seq_length=64, train_split=0.9):  # 减小batch_size和序列长度
    """Load and prepare Shakespeare data for training."""
    # Load text data
    text = load_shakespeare_data()
    
    # Create character mappings
    char_to_idx, idx_to_char, vocab_size = create_char_mappings(text)
    
    # 增加batch_size以充分利用多GPU，但要考虑显存限制
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        # 使用较小的每GPU batch size
        batch_size = batch_size * num_gpus
        print(f"使用 {num_gpus} 个GPU, 总batch_size调整为: {batch_size}")
    
    # Calculate split point
    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Create datasets
    train_dataset = ShakespeareDataset(train_text, char_to_idx, seq_length)
    val_dataset = ShakespeareDataset(val_text, char_to_idx, seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, char_to_idx, idx_to_char, vocab_size


def create_masks(src, tgt=None):
    """Create masks for transformer input."""
    # Source padding mask (not needed for character-level, but included for completeness)
    src_mask = torch.ones((src.size(0), 1, 1, src.size(1))).bool()
    
    if tgt is None:
        return src_mask
    
    # Target padding mask
    tgt_mask = torch.ones((tgt.size(0), 1, 1, tgt.size(1))).bool()
    
    # Look-ahead mask
    seq_length = tgt.size(1)
    look_ahead_mask = ~torch.triu(torch.ones((seq_length, seq_length)), diagonal=1).bool()
    
    # Combine masks
    combined_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)
    
    return src_mask, combined_mask


def generate_text(model, start_text, char_to_idx, idx_to_char, seq_length, max_length=1000):
    """Generate text using the trained model."""
    model.eval()
    current_chars = list(start_text)[-seq_length:]
    generated_chars = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # Convert current characters to indices
            current_seq = torch.tensor([char_to_idx[c] for c in current_chars]).unsqueeze(0)
            
            # Generate prediction
            logits = model(current_seq)
            probs = torch.softmax(logits[0, -1], dim=0)
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_char_idx]
            
            # Append to generated text
            generated_chars.append(next_char)
            
            # Update current sequence
            current_chars = current_chars[1:] + [next_char]
    
    return ''.join(generated_chars)
