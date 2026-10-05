# -*- coding: utf-8 -*-
"""toy_gpt/model.py —— 0.21M 字符级 GPT（项目被测模型，Day 2 自研）
Head/MHA/FFN/Block/GPT 与训练脚本同构；Block 支持 ln_mode(two/shared/entrance)
与 norm_pos(pre/post) 以支撑架构对照实验（见 experiments/）。
参数量手推对账：209,729 ≈ 12·d²（emb 65×64 + 4 层 + pos 表 + lnf/head）。"""
import torch, torch.nn as nn, torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embd, hs, block_size):
        super().__init__(); self.hs = hs
        self.key   = nn.Linear(n_embd, hs, bias=False)
        self.query = nn.Linear(n_embd, hs, bias=False)
        self.value = nn.Linear(n_embd, hs, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2, -1) * self.hs ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)


class MHA(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, n_embd // n_head, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(), nn.Linear(4*n_embd, n_embd))
    def forward(self, x): return self.net(x)


class Block(nn.Module):
    """ln_mode: 'two'(标准) / 'shared' / 'entrance'；norm_pos: 'pre' / 'post'"""
    def __init__(self, n_embd, n_head, block_size, ln_mode='two', norm_pos='pre'):
        super().__init__()
        assert not (norm_pos == 'post' and ln_mode != 'two')
        self.ln_mode, self.norm_pos = ln_mode, norm_pos
        self.sa = MHA(n_embd, n_head, block_size); self.ffwd = FFN(n_embd)
        if ln_mode == 'shared':
            self.ln = nn.LayerNorm(n_embd)
        else:
            self.ln1 = nn.LayerNorm(n_embd)
            if ln_mode == 'two':
                self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        if self.norm_pos == 'pre':
            if self.ln_mode == 'two':
                x = x + self.sa(self.ln1(x)); x = x + self.ffwd(self.ln2(x))
            elif self.ln_mode == 'shared':
                x = x + self.sa(self.ln(x));   x = x + self.ffwd(self.ln(x))
            else:
                x = x + self.sa(self.ln1(x)); x = x + self.ffwd(x)
        else:
            x = self.ln1(x + self.sa(x));     x = self.ln2(x + self.ffwd(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer=4, n_embd=64, n_head=4, block_size=32,
                 ln_mode='two', norm_pos='pre'):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, ln_mode, norm_pos)
                                      for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.blocks(x); x = self.lnf(x)
        logits = self.head(x)
        loss = None if targets is None else \
            F.cross_entropy(logits.view(B*T, -1), targets.reshape(B*T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n_tok, block_size=32):
        for _ in range(n_tok):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx
