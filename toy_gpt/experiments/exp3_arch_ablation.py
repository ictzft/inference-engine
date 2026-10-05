# -*- coding: utf-8 -*-
"""
Day 3 实验作业（自由实验日）—— 三个实验，全部围绕 Day 2 的悬案

用法（在任意目录运行都可以，数据文件按脚本所在目录查找）：
    python day3-实验作业.py A        # 实验A：LN 三方案 × 深度（8 层 / 12 层）
    python day3-实验作业.py B        # 实验B：post-norm 三连（4层 → 8层 → warmup 救援）
    python day3-实验作业.py C        # 实验C：等参数预算 深瘦 vs 浅胖
    python day3-实验作业.py all      # 全部（约 45 分钟 GPU）
    末尾加 quick 用小参数快速自检（例如：python day3-实验作业.py A quick）

仪式（每个实验都一样，来自 Day 2 的传统）：
    ① 先在下面"预测"处写下你的预测 → ② 跑 → ③ 在"实测"处记录 → ④ 写一句解释。
    预测错了收获最大。
"""
import sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
QUICK = 'quick' in sys.argv

# ---------------- 数据（与 Day 2 完全一致） ----------------
# 坑：相对路径是按"运行时所在目录"找的，不是按脚本所在目录。VS Code 直接点运行时
# 当前目录往往是工作区目录，所以这里锚定到脚本自己所在的目录。
BASE = Path(__file__).resolve().parent.parent
text = open(BASE / 'input.txt', encoding='utf-8').read()
chars = sorted(set(text)); vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

batch_size, block_size = 32, 32
max_iters  = 30   if QUICK else 5000
eval_iters = 5    if QUICK else 200
lr = 1e-3

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]
def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ---------------- 模型（参数显式传入——Day 2 坑3 的教训：不用全局变量） ----------------
class Head(nn.Module):
    def __init__(self, n_embd, hs):
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
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, n_embd // n_head) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(), nn.Linear(4*n_embd, n_embd))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    """ln_mode: 'two'(标准) / 'shared'(同一个LN用两次) / 'entrance'(只装入口一个)
       norm_pos: 'pre' / 'post'"""
    def __init__(self, n_embd, n_head, ln_mode='two', norm_pos='pre'):
        super().__init__()
        assert not (norm_pos == 'post' and ln_mode != 'two'), \
            'post 分支固定用 ln1+ln2，norm_pos=post 时 ln_mode 必须是 two'
        self.ln_mode, self.norm_pos = ln_mode, norm_pos
        self.sa = MHA(n_embd, n_head); self.ffwd = FFN(n_embd)
        if ln_mode == 'shared':
            self.ln = nn.LayerNorm(n_embd)
        else:
            self.ln1 = nn.LayerNorm(n_embd)
            if ln_mode == 'two':
                self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        if self.norm_pos == 'pre':                     # pre-norm 家族
            if self.ln_mode == 'two':                  # A 标准版
                x = x + self.sa(self.ln1(x)); x = x + self.ffwd(self.ln2(x))
            elif self.ln_mode == 'shared':             # B 共享版
                x = x + self.sa(self.ln(x));   x = x + self.ffwd(self.ln(x))
            else:                                      # C 只装入口（FFN 裸读）
                x = x + self.sa(self.ln1(x)); x = x + self.ffwd(x)
        else:                                          # post-norm 家族（实验B）
            x = self.ln1(x + self.sa(x));     x = self.ln2(x + self.ffwd(x))
        return x

class GPT(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, ln_mode='two', norm_pos='pre'):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, ln_mode, norm_pos) for _ in range(n_layer)])
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
    def generate(self, idx, n_tok):
        for _ in range(n_tok):
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

def train(tag, n_layer, n_embd=64, n_head=4, ln_mode='two', norm_pos='pre', warmup=0):
    torch.manual_seed(1337)
    m = GPT(n_layer, n_embd, n_head, ln_mode, norm_pos).to(device)
    params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    t0 = time.time()
    final = {}
    for it in range(max_iters):
        if warmup > 0:   # warmup：前 warmup 步 lr 从 0 线性升到目标值
            for g in opt.param_groups:
                g['lr'] = lr * min(1.0, (it + 1) / warmup)
        if (it + 1) % (max_iters // 5 if QUICK else 1000) == 0 or it == 0:
            L = estimate(m)
            final = L
            print(f'  [{tag}] step {it+1}: train {L["train"]:.4f}, val {L["val"]:.4f}')
        xb, yb = get_batch('train')
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    dt = time.time() - t0
    m.eval()
    return m, params, final, dt

def gen_speed(m, n_tok=300, repeat=3):
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    ts = []
    for _ in range(repeat):
        t0 = time.time(); m.generate(ctx, n_tok); ts.append((time.time() - t0) / n_tok * 1000)
    return sum(ts) / len(ts)

def record(label):
    print(f'\n--- 你的记录（{label}）---')
    print('预测：')
    print('实测：')
    print('解释：')

# ============================================================
# 实验A：LN 三方案 × 深度——验证"架构卫生随深度兑现"（Day 2 ⑭ 悬案）
# 对照基准：4 层时三方案全是 ~1.76（day2-两个LN对照实验.py）
# ============================================================
def exp_A():
    print('=' * 60)
    print('实验A：LN 三方案（two/shared/entrance）× 深度（8 层、12 层）')
    print('预测提示：8/12 层时三方案还会全一样吗？谁先掉队？差距多大？')
    print('=' * 60)
    depths = [8] if QUICK else [8, 12]
    results = {}
    for depth in depths:
        for mode in ['two', 'shared', 'entrance']:
            m, p, final, dt = train(f'{mode}-{depth}层', depth, ln_mode=mode)
            results[(depth, mode)] = (final.get('val', float('nan')), dt)
            del m; torch.cuda.empty_cache()
    print('\n===== 汇总（val loss）=====')
    print(f'{"":<12}{"two(标准)":>12}{"shared(共享)":>14}{"entrance(单LN)":>16}   （4层基准：全≈1.76）')
    for depth in depths:
        row = ' '.join(f'{results[(depth, m_)][0]:>{w}.4f}' for m_, w in
                       [('two', 12), ('shared', 14), ('entrance', 16)])
        print(f'{depth}层{"":<8}{row}')
    record('实验A')

# ============================================================
# 实验B：post-norm 三连（Day 2 ⑰ 的验证作业）
# ============================================================
def exp_B():
    print('=' * 60)
    print('实验B：post-norm 三连（对照：pre-norm 4 层 = 1.76）')
    print('预测提示：post 4 层能到多少？8 层会更差还是训崩？warmup 能救回多少？')
    print('=' * 60)
    m, p, f1, dt1 = train('post-4层', 4, norm_pos='post')
    del m; torch.cuda.empty_cache()
    m, p, f2, dt2 = train('post-8层', 8, norm_pos='post')
    del m; torch.cuda.empty_cache()
    wu = 20 if QUICK else 1000
    m, p, f3, dt3 = train(f'post-8层+warmup{wu}', 8, norm_pos='post', warmup=wu)
    del m; torch.cuda.empty_cache()
    print('\n===== 汇总 =====')
    print(f'pre-4层基准 1.76 | post-4层 {f1.get("val", float("nan")):.4f} | '
          f'post-8层 {f2.get("val", float("nan")):.4f} | post-8层+warmup {f3.get("val", float("nan")):.4f}')
    record('实验B')

# ============================================================
# 实验C：等参数预算 深瘦 vs 浅胖（Day 2 ㉕ 四堵墙的实证）
# ============================================================
def exp_C():
    print('=' * 60)
    print('实验C：等参数预算 深瘦(64层×64维) vs 浅胖(4层×256维)，各约 3.2M')
    print('预测提示：val 谁低？训练谁快？每字生成谁快（回想"kernel 启动开销"）？')
    print('=' * 60)
    deep_l, fat_e = (16, 96) if QUICK else (64, 256)
    m1, p1, f1, dt1 = train(f'深瘦{deep_l}层×64维', deep_l, n_embd=64, n_head=4)
    s1 = gen_speed(m1)
    del m1; torch.cuda.empty_cache()
    m2, p2, f2, dt2 = train(f'浅胖4层×{fat_e}维', 4, n_embd=fat_e, n_head=4)
    s2 = gen_speed(m2)
    del m2; torch.cuda.empty_cache()
    print('\n===== 汇总 =====')
    print(f'{"":<12}{"参数量":>10}{"val":>8}{"训练耗时":>10}{"ms/字":>8}')
    print(f'深瘦{deep_l}×64{"":<4}{p1:>10,}{f1.get("val", float("nan")):>8.4f}{dt1:>9.0f}s{s1:>8.1f}')
    print(f'浅胖4×{fat_e}{"":<6}{p2:>10,}{f2.get("val", float("nan")):>8.4f}{dt2:>9.0f}s{s2:>8.1f}')
    record('实验C')

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    todo = {'A': [exp_A], 'B': [exp_B], 'C': [exp_C],
            'all': [exp_A, exp_B, exp_C]}.get(which)
    if todo is None:
        print(__doc__); sys.exit(1)
    for fn in todo:
        fn()
    print('\n全部完成。记得把三份"预测/实测/解释"誊进 day2-笔记.md 或明天的自述文档。')
