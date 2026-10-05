# -*- coding: utf-8 -*-
"""对照实验：一个 Block 为什么要两个 LayerNorm？
方案A：ln1、ln2 各一个（标准设计）
方案B：同一个 LN 对象调用两次（两次都标准化，但 γ/β 被迫共享）
方案C：只装一个入口 LN（注意力读标准化流，FFN 裸读加过增量的流）
另附：流统计探针——训练好的模型里，干道两个时刻的量级、ln1/ln2 的"口味"差异
"""
import torch, torch.nn as nn, torch.nn.functional as F, time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from pathlib import Path
text = open(Path(__file__).resolve().parent.parent / 'input.txt', encoding='utf-8').read()
chars = sorted(set(text)); vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]

batch_size, block_size = 32, 32
n_embd, n_head = 64, 4
max_iters, eval_iters, lr = 5000, 200, 1e-3

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
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd // n_head) for _ in range(n_head)])
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
    def __init__(self, mode='two'):
        super().__init__()
        assert mode in ('two', 'shared', 'entrance')
        self.mode = mode
        self.sa = MHA(); self.ffwd = FFN()
        if mode == 'shared':
            self.ln = nn.LayerNorm(n_embd)              # 一个对象，两处调用
        else:
            self.ln1 = nn.LayerNorm(n_embd)
            if mode == 'two':
                self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        if self.mode == 'two':                          # A：标准
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        elif self.mode == 'shared':                     # B：共享同一个 LN
            x = x + self.sa(self.ln(x))
            x = x + self.ffwd(self.ln(x))
        else:                                           # C：FFN 裸读
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(x)
        return x

class GPT(nn.Module):
    def __init__(self, mode='two', n_layer=4):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(mode) for _ in range(n_layer)])
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

@torch.no_grad()
def estimate(m):
    out = {}; m.eval()
    for sp in ['train', 'val']:
        ls = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(sp); _, l = m(X, Y); ls[k] = l.item()
        out[sp] = ls.mean()
    m.train(); return out

def train(mode, tag):
    torch.manual_seed(1337)
    m = GPT(mode).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    t0 = time.time(); ok = True
    for it in range(max_iters):
        if (it + 1) % 1000 == 0:
            L = estimate(m)
            print(f'  [{tag}] step {it+1}: train {L["train"]:.4f}, val {L["val"]:.4f}')
            if torch.isnan(L['val']) or torch.isinf(L['val']):
                ok = False; break
        xb, yb = get_batch('train')
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    print(f'  [{tag}] 耗时 {time.time()-t0:.0f}s' + ('' if ok else '  ← 训崩(NaN)，提前终止'))
    return m

print('== 三方案训练对比（4 层，同种子 1337，5000 步）==')
mA = train('two',      'A·标准两个LN ')
mB = train('shared',   'B·同一LN用两次')
mC = train('entrance', 'C·只装入口一个')

# ---------- 流统计探针：用训练好的标准模型 ----------
print('\n== 流统计探针（训练后的标准模型 A，一个真实 batch）==')
print('  （std = 平均每个 token 的 64 维特征的标准差，衡量"流当时的量级"）')
mA.eval()
with torch.no_grad():
    X, Y = get_batch('train')
    B, T = X.shape
    h = mA.tok(X) + mA.pos(torch.arange(T, device=device))
    print(f'  {"层":<4}{"干道x0(注意力读前)":>20}{"x1=+Δattn后(FFN读前)":>22}{"ln1递给注意力":>16}{"ln2递给FFN":>14}')
    for i, blk in enumerate(mA.blocks):
        s0 = h.std(dim=-1).mean().item()
        attn_read = blk.ln1(h)
        s_l1 = attn_read.std(dim=-1).mean().item()
        x1 = h + blk.sa(attn_read)
        s1 = x1.std(dim=-1).mean().item()
        ffn_read = blk.ln2(x1)
        s_l2 = ffn_read.std(dim=-1).mean().item()
        h = x1 + blk.ffwd(ffn_read)
        print(f'  {i:<4}{s0:>20.3f}{s1:>22.3f}{s_l1:>16.3f}{s_l2:>14.3f}')

    print('\n  ln1.γ 与 ln2.γ 的余弦相似度（1.0=同口味，0=毫无关系）：')
    for i, blk in enumerate(mA.blocks):
        cos = F.cosine_similarity(blk.ln1.weight, blk.ln2.weight, dim=0).item()
        print(f'    Block{i}: {cos:+.3f}')
