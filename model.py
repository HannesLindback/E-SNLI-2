"""Module implementing a transformer encoder-decoder model for prediciting explanations and labels.

Based on the An even more annotated transformer tutorial:
https://pi-tau.github.io/posts/transformer/"""


import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class PredictLabel(nn.Module):
    """A module consisting of a transformer encoder and a three-layer MLP.
    
    Used for predicting labels."""
    
    def __init__(self, vocab_size, max_seq_len,
            d_model, n_heads, n_enc_layers, dim_ff, dropout,
            n_labels):
        super(PredictLabel, self).__init__()
        
        self.d_model = d_model
        scale = np.sqrt(d_model)
        pos_embed = nn.Parameter(torch.randn(max_seq_len, d_model))
        word_embed = nn.Parameter(torch.randn(vocab_size, d_model) / scale)

        self.embedding = TokenEmbedding(word_embed, pos_embed, scale, dropout)
        
        self.encoder_stack = nn.ModuleList((
            EncoderBlock(d_model, n_heads, dim_ff, dropout) for _ in range(n_enc_layers)
        ))
        self.enc_norm = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(max_seq_len*d_model, dim_ff),
            nn.Linear(dim_ff, dim_ff),
            nn.Linear(dim_ff, n_labels))
        
    def encode(self, src, src_mask):
        for encoder in self.encoder_stack:
            src = encoder(src, src_mask)
        return self.enc_norm(src)
    
    def forward(self, x, source_mask):
        B, _, = x.shape
        embedded = self.embedding(x)
        encoded = self.encode(embedded, source_mask)
        output = self.mlp(encoded.view(B, -1))
        return output
        
    @torch.no_grad()
    def greedy_decode(self, src, src_mask):
        B, _, = src.shape
        self.eval()
        embedded = self.embedding(src)
        encoded = self.encode(embedded, src_mask)            
        label = torch.argmax(self.mlp(encoded.view(B, -1)))

        return label

class EncoderDecoder(nn.Module):
    """A transformer encoder-decoder model."""
    
    def __init__(
            self, src_vocab_size, tgt_vocab_size, max_seq_len,
            d_model, n_heads, n_enc_layers, n_dec_layers, dim_ff, dropout):
        
        super(EncoderDecoder, self).__init__()
        
        self.d_model = d_model
        scale = np.sqrt(d_model)
        pos_embed = nn.Parameter(torch.randn(max_seq_len, d_model))
        src_word_embed = nn.Parameter(torch.randn(src_vocab_size, d_model) / scale)
        
        if tgt_vocab_size is None:
            tgt_word_embed = src_word_embed
        else:
            tgt_word_embed = nn.Parameter(torch.randn(tgt_vocab_size, d_model) / scale)
            
        self.src_embed = TokenEmbedding(src_word_embed, pos_embed, scale, dropout)
        self.tgt_embed = TokenEmbedding(tgt_word_embed, pos_embed, scale, dropout)
        
        self.encoder_stack = nn.ModuleList((
            EncoderBlock(d_model, n_heads, dim_ff, dropout) for _ in range(n_enc_layers)
        ))
        self.enc_norm = nn.LayerNorm(d_model)
        
        self.decoder_stack = nn.ModuleList((
            DecoderBlock(d_model, n_heads, dim_ff, dropout) for _ in range(n_dec_layers)
        ))
        self.dec_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(in_features=d_model, out_features=tgt_vocab_size)
        
    def encode(self, src, src_mask):
        for encoder in self.encoder_stack:
            src = encoder(src, src_mask)
        return self.enc_norm(src)

    def decode(self, tgt, mem, mem_mask):
        for decoder in self.decoder_stack:
            tgt = decoder(tgt, mem, mem_mask)
        return self.dec_norm(tgt)

    def forward(self, u, v, tgt, u_mask, v_mask):
        """Takes sequences u and v as input and returns logits over most likely output sequence."""
        
        u = self.src_embed(u)
        u = self.encode(u, u_mask)
        v = self.src_embed(v)
        v = self.encode(v, v_mask)
        f = torch.concat((u, v), dim=1)
        f_mask = torch.concat((u_mask, v_mask), dim=1)
        tgt = self.tgt_embed(tgt)
        out = self.decode(tgt, f, mem_mask=f_mask)
        tgt_scores = self.linear(out)
        return tgt_scores
    
    @torch.no_grad()
    def greedy_decode(self, u, v, u_mask, v_mask, bos_idx, eos_idx, max_len=64):
        """Greedily decodes the most likely target sequence one token at a time
        from an input of two sequences u and v."""
        
        B = u.shape[0]
            
        done = {i : False for i in range(B)}
        was_training = self.training
        self.eval()

        tgt = torch.LongTensor([[bos_idx]] * B).to(u.device)
        u = self.src_embed(u)
        u = self.encode(u, u_mask)
        v = self.src_embed(v)
        v = self.encode(v, v_mask)
        mem = torch.concat((u, v), dim=1)
        mem_mask = torch.concat((u_mask, v_mask), dim=1)
        
        for _ in range(max_len-1):
            tgt_emb = self.tgt_embed(tgt)
            out = self.decode(tgt_emb, mem, mem_mask=mem_mask)
            scores = self.linear(out)
            next_idx = torch.max(scores[:, -1:], dim=-1).indices
            tgt = torch.concat((tgt, next_idx), dim=1)

            for i, idx in enumerate(next_idx):
                if idx[0] == eos_idx: done[i] = True
            if False not in done.values(): break
            
        if was_training: self.train()
        return tgt
    
    
class EncoderBlock(nn.Module):
    """A transformer encoder block. Uses pre-layer add and normalization."""
    
    def __init__(self, d_model, n_heads, dim_mlp, dropout):
        super(EncoderBlock, self).__init__()

        self.attention      = MultiHeadAttention(in_dim=d_model,
                                            qk_dim=d_model,
                                            v_dim=d_model,
                                            n_heads=n_heads,
                                            out_dim=d_model,
                                            attn_dropout=dropout)
        
        self.ff             = FeedForward(d_model=d_model,
                                          d_ff=dim_mlp,
                                          dropout=dropout)
        
        self.attention_norm    = nn.LayerNorm(d_model)
        self.ff_norm           = nn.LayerNorm(d_model)
        self.attention_dropout = nn.Dropout(p=dropout)
        self.ff_dropout        = nn.Dropout(p=dropout)
            
    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(dim=-1)
            
        x = self.attention_norm(x)
        z, attention_weights = self.attention(x, x, x, mask=mask)
        z = x + self.attention_dropout(z)
        
        z = self.ff_norm(z)
        r = self.ff(z)
        r = z + self.ff_dropout(r)
        return r
    
    
class DecoderBlock(nn.Module):
    """A transformer decoder block. Uses pre-layer add and normalization."""

    def __init__(self, d_model, n_heads, dim_mlp, dropout):
        super(DecoderBlock, self).__init__()
        
        self.masked_attention       = MultiHeadAttention(in_dim=d_model,
                                            qk_dim=d_model,
                                            v_dim=d_model,
                                            n_heads=n_heads,
                                            out_dim=d_model,
                                            attn_dropout=dropout)
        self.cross_attention        = MultiHeadAttention(in_dim=d_model,
                                            qk_dim=d_model,
                                            v_dim=d_model,
                                            n_heads=n_heads,
                                            out_dim=d_model,
                                            attn_dropout=dropout)
        self.ff                     = FeedForward(d_model=d_model,
                                                  d_ff=dim_mlp,
                                                  dropout=dropout)
        self.masked_attention_norm   = nn.LayerNorm(d_model)
        self.ff_norm          = nn.LayerNorm(d_model)
        self.cross_attention_norm = nn.LayerNorm(d_model)
        self.masked_attention_dropout = nn.Dropout(p=dropout)
        self.cross_attention_dropout = nn.Dropout(p=dropout)
        self.ff_dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, mem, mem_mask=None):
        _, T, _ = x.shape
        causal_mask = torch.ones(1, T, T, dtype=torch.bool).tril().to(x.device)
        x = self.masked_attention_norm(x)
        z, attn_weights = self.masked_attention(x, x, x, mask=causal_mask)
        z = x + self.masked_attention_dropout(z)

        if mem_mask is not None:
            mem_mask = mem_mask.unsqueeze(dim=-2)
            
        z = self.cross_attention_norm(z)
        c, cross_attn_weights = self.cross_attention(z, mem, mem, mask=mem_mask)
        c = z + self.cross_attention_dropout(c)

        c = self.ff_norm(c)
        r = self.ff(c)
        r = c + self.ff_dropout(r)

        return r


class TokenEmbedding(nn.Module):
    """Class for creating a positional and token embedding for an input."""
    
    def __init__(self, word_embed_weight, pos_embed_weight, scale, dropout):
        super().__init__()
        max_len, _ = pos_embed_weight.shape
        self.max_len = max_len
        self.word_embed_weight = word_embed_weight
        self.pos_embed_weight = pos_embed_weight
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("positions", torch.arange(max_len).unsqueeze(dim=0))
        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([scale])))

    def forward(self, x):
        _, T = x.shape
        if T > self.max_len:
            raise RuntimeError("Sequence length exceeds the maximum allowed limit")

        pos = self.positions[:, :T]
        word_embed = F.embedding(x, self.word_embed_weight)
        pos_embed = F.embedding(pos, self.pos_embed_weight)
        embed = pos_embed + word_embed * self.scale
        return self.dropout(embed) 


class MultiHeadAttention(nn.Module):
    """Class for implementing multi-headed attention for a transformer model."""
    
    def __init__(self, in_dim, qk_dim, v_dim, n_heads, attn_dropout=0.):
        super(MultiHeadAttention, self).__init__()
        
        self.Q_linear = nn.Linear(in_dim, qk_dim, bias=False)
        self.K_linear = nn.Linear(in_dim, qk_dim, bias=False)
        self.V_linear = nn.Linear(in_dim, v_dim, bias=False)
        self.Wo = nn.Linear(v_dim, in_dim)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(p=attn_dropout)
        self.dk = qk_dim
        self.softmax = nn.Softmax(dim=-1)
        
        nn.init.normal_(self.Q_linear.weight,
                        std=np.sqrt(2 / (in_dim + qk_dim//n_heads)))
        nn.init.normal_(self.K_linear.weight,
                        std=np.sqrt(2 / (in_dim + qk_dim//n_heads)))
        nn.init.zeros_(self.Wo.bias)
        
    def forward(self, queries, keys, values, mask=None):
        batch_size, n_tokens, emb_dim = queries.shape
        batch_size, n_tokens_kv, emb_dim = keys.shape
        
        if mask is not None:
            mask = mask.unsqueeze(dim=1)
            
        Q = self.Q_linear(queries).view(batch_size, n_tokens, self.n_heads, -1).transpose(1, 2)
        K = self.K_linear(keys).view(batch_size, n_tokens_kv, self.n_heads, -1).transpose(1, 2)
        V = self.V_linear(values).view(batch_size, n_tokens_kv, self.n_heads, -1).transpose(1, 2)
        
        attention = torch.matmul(Q, K.transpose(2,3))
        scalar = math.sqrt(self.dk)
        scaled_QK = torch.div(attention, scalar)
        
        if mask is not None:
            mask_value = -6.5504e4 if scaled_QK.dtype == torch.float16 else -1e9
            scaled_QK.masked_fill_(~mask, mask_value)
        
        attention = self.softmax(scaled_QK)
        attn_weights = self.dropout(attention)
        
        attention = torch.matmul(attn_weights, V)
        attention = attention.transpose(1, 2).reshape(batch_size, n_tokens, -1)
        
        return self.Wo(attention), attn_weights
    

class FeedForward(nn.Module):
    """Implements the feedforward sublayer in a transformer model."""
    
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.linear1    = nn.Linear(d_model, d_ff)
        self.relu       = nn.ReLU()
        self.linear2    = nn.Linear(d_ff, d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.dropout(x)
