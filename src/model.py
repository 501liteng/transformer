import torch
import torch.nn as nn
import math
from typing import Optional


# Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.depth = embedding_dim // num_heads

        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)
        self.dense = nn.Linear(embedding_dim, embedding_dim)

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.split_heads(self.W_Q(Q), batch_size)
        K = self.split_heads(self.W_K(K), batch_size)
        V = self.split_heads(self.W_V(V), batch_size)

        attention, _ = scaled_dot_product_attention(Q, K, V, mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        output = self.dense(attention)

        return output


# FeedForward Network
class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, ffn_dim):
        super(FeedForwardNetwork, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embedding_dim)
        )

    def forward(self, x):
        return self.ffn(x)


# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = FeedForwardNetwork(embedding_dim, ffn_dim)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Multi-Head Attention
        attn_output = self.multi_head_attention(x, x, x, mask)
        x = self.layer_norm_1(x + self.dropout(attn_output))

        # Feed-Forward Network
        ffn_output = self.ffn(x)
        x = self.layer_norm_2(x + self.dropout(ffn_output))

        return x


# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.cross_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = FeedForwardNetwork(embedding_dim, ffn_dim)
        
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.layer_norm_3 = nn.LayerNorm(embedding_dim)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        # Self Attention
        attn1_output = self.self_attention(x, x, x, look_ahead_mask)
        x = self.layer_norm_1(x + self.dropout(attn1_output))
        
        # Cross Attention
        attn2_output = self.cross_attention(x, enc_output, enc_output, padding_mask)
        x = self.layer_norm_2(x + self.dropout(attn2_output))
        
        # Feed Forward Network
        ffn_output = self.ffn(x)
        x = self.layer_norm_3(x + self.dropout(ffn_output))
        
        return x


# Transformer Model with Encoder and Decoder
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, num_heads, num_layers, ffn_dim):
        super(Transformer, self).__init__()
        
        # Encoder
        self.encoder = TransformerEncoder(src_vocab_size, embedding_dim, num_heads, num_layers, ffn_dim)
        
        # Decoder
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, embedding_dim))
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Linear(embedding_dim, tgt_vocab_size)
        
    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones((size, size)), diagonal=1)
        return mask == 0
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        enc_output = self.encoder(src, src_mask)
        
        # Decoder
        tgt_emb = self.decoder_embedding(tgt)
        tgt_emb = tgt_emb + self.positional_encoding[:, :tgt_emb.size(1), :]
        
        # Create look-ahead mask for decoder self-attention
        look_ahead_mask = self.create_look_ahead_mask(tgt.size(1)).to(tgt.device)
        if tgt_mask is not None:
            look_ahead_mask = look_ahead_mask & tgt_mask
            
        # Pass through decoder layers
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, look_ahead_mask, src_mask)
            
        # Final linear layer
        output = self.final_layer(dec_output)
        
        return output


# Transformer Encoder Model
class TransformerEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, ffn_dim, use_positional_encoding=True):
        super(TransformerEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding = torch.nn.Parameter(torch.zeros(1, 1000, embedding_dim))
        self.encoder_layers = torch.nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        if self.use_positional_encoding:
            x = x + self.positional_encoding[:, :x.size(1), :]
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x
