# -*- coding: utf-8 -*-
"""对照实验：去掉残差的 Block 会怎样（4 层 / 8 层 vs 昨天有残差的 1.76）"""
import torch, torch.nn as nn, torch.nn.functional as F, time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from pathlib import Path
text = open(Path(__file__).resolve().parent.parent / 'input.txt', encoding='utf-8').read()
chars = sorted(set(text)); vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

batch_size, block_size = 32, 32
n_embd, n_head, dropout = 64, 4, 0.0
max_iters, eval_interval, eval_iters, lr = 5000, 500, 200, 1e-3

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]
def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class Head(nn.Module):
    def __init__(self, hs):
        super().__init__(); self.hs = hs
        self.key = nn.Linear(n_embd, hs, bias=False)
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
    def __init__(self, nh, hs):
        super().__init__()
        self.heads = nn.ModuleList([Head(hs) for _ in range(nh)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(),
                                 nn.Linear(4*n_embd, n_embd))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, residual=True):
        super().__init__(); self.residual = residual
        self.sa = MHA(n_head, n_embd // n_head)
        self.ffwd = FFN()
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        if self.residual:
            x = x + self.sa(self.ln1(x))   # 有残差：读流 → 写增量
            x = x + self.ffwd(self.ln2(x))
        else:
            x = self.sa(self.ln1(x))       # 无残差：全量重写
            x = self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, n_layer=4, residual=True):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(residual) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.blocks(x); x = self.lnf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.reshape(B*T))
        return logits, loss
    def generate(self, idx, n):
        for _ in range(n):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx

@torch.no_grad()
def estimate(m):
    out = {}; m.eval()
    for sp in ['train', 'val']:
        ls = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(sp); _, l = m(X, Y); ls[k] = l.item()
        out[sp] = ls.mean()
    m.train(); return out

# ---------- 梯度探针：初始化时，同一 batch，看梯度能否到达浅层 ----------
torch.manual_seed(42)
XB, YB = get_batch('train')          # 四个探针共用同一个 batch

def grad_probe(residual, n_layer):
    torch.manual_seed(1337)
    m = GPT(n_layer, residual).to(device)
    _, loss = m(XB, YB)
    m.zero_grad(set_to_none=True); loss.backward()
    norms = []
    for blk in m.blocks:
        g2 = sum(p.grad.norm().item() ** 2 for p in blk.parameters() if p.grad is not None)
        norms.append(round(g2 ** 0.5, 4))
    return norms

print('== 梯度探针（初始化时，各 Block 参数的梯度范数，同一 batch）==')
print('  有残差 4层:', grad_probe(True, 4))
print('  无残差 4层:', grad_probe(False, 4))
print('  有残差 8层:', grad_probe(True, 8))
print('  无残差 8层:', grad_probe(False, 8))

# ---------- 训练对照 ----------
def train(n_layer, residual, tag):
    torch.manual_seed(1337)
    m = GPT(n_layer, residual).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    t0 = time.time()
    for it in range(max_iters):
        if it == 0 or (it + 1) % 1000 == 0:
            L = estimate(m)
            print(f'  [{tag}] step {it+1}: train {L["train"]:.4f}, val {L["val"]:.4f}')
        xb, yb = get_batch('train')
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    dt = time.time() - t0
    m.eval()
    sample = decode(m.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 200)[0].tolist())
    return dt, sample

print('\n== 训练：无残差 4 层（对照：有残差 4 层 = train 1.57 / val 1.76）==')
dt4, s4 = train(4, False, '无残差4层')
print(f'  耗时 {dt4:.0f}s；生成样例：')
print(' ', repr(s4[:160]))

print('\n== 训练：无残差 8 层 ==')
dt8, s8 = train(8, False, '无残差8层')
print(f'  耗时 {dt8:.0f}s；生成样例：')
print(' ', repr(s8[:160]))
