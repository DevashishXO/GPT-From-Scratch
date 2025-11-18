import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import TemplateProcessing

# ---------------------
# Reproducibility
# ---------------------
torch.manual_seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------
# Config
# ---------------------
@dataclass
class Config:
    # data
    dataset_path: str = "dataset/new_input.txt"
    tokenizer_dir: str = "tokenizer"
    tokenizer_file: str = "tokenizer/bpe_tokenizer.json"
    vocab_size: int = 8000 # changed from 10000
    force_retrain_tokenizer: bool = False  # set True to retrain with ByteLevel BPE
    # model
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 4 # changed from 6
    dropout: float = 0.4           # stronger regularization
    emb_dropout: float = 0.1       # new: embedding dropout
    block_size: int = 256
    # training
    batch_size: int = 32 # changed from 64
    max_iters: int = 25000
    eval_interval: int = 1000 # changed from 250
    eval_iters: int = 50 # changed from 200
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 2000
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    grad_accum_steps: int = 2 # changed from 1
    use_amp: bool = True
    patience_evals: int = 6        # new: early stopping patience
    save_every: int = 500
    ckpt_dir: str = "checkpoints"
    resume_path: Optional[str] = None
    # loss
    label_smoothing: float = 0.1   # new: improves generalization
    # generation
    gen_max_new_tokens: int = 700
    gen_temperature: float = 1.0
    gen_top_k: int = 50

cfg = Config()
# === [NEW] Set up Drive output directories ===
import datetime
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
drive_root = f"/content/drive/MyDrive/gpt_from_scratch_runs/run_{run_id}"

cfg.ckpt_dir = os.path.join(drive_root, "checkpoints")
cfg.tokenizer_dir = os.path.join(drive_root, "tokenizer")
cfg.tokenizer_file = os.path.join(cfg.tokenizer_dir, "bpe_tokenizer.json")
os.makedirs(cfg.ckpt_dir, exist_ok=True)
os.makedirs(cfg.tokenizer_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------
# Clone dataset repo (COLAB)
# ---------------------
if not os.path.exists(cfg.dataset_path):
    print("📥 Cloning dataset repo...")
    os.system("git clone https://github.com/DevashishXO/GPT-From-Scratch.git")
    os.chdir("GPT-From-Scratch")
else:
    print("✅ Dataset already exists.")

# ---------------------
# Tokenizer setup
# ---------------------
os.makedirs(cfg.tokenizer_dir, exist_ok=True)

def train_or_load_tokenizer():
    tok_path = cfg.tokenizer_file
    if cfg.force_retrain_tokenizer and os.path.exists(tok_path):
        print("♻️ Forcing tokenizer retrain, removing old file...")
        try:
            os.remove(tok_path)
        except FileNotFoundError:
            pass

    if not os.path.exists(tok_path):
        print("🔧 Training new BPE tokenizer...")
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        # ByteLevel is more robust for multilingual/raw punctuation; switch if retraining
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel() if cfg.force_retrain_tokenizer else pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=cfg.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        )
        tokenizer.train([cfg.dataset_path], trainer)
        if isinstance(tokenizer.model, models.BPE):
            tokenizer.model.unk_token = "[UNK]"
        bos_id = tokenizer.token_to_id("[BOS]")
        eos_id = tokenizer.token_to_id("[EOS]")
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [EOS] [BOS] $B [EOS]",
            special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)]
        )
        tokenizer.save(tok_path)
        print("✅ Tokenizer trained and saved.")
    else:
        print("✅ Existing tokenizer found. Loading...")
    tokenizer = Tokenizer.from_file(tok_path)
    if isinstance(tokenizer.model, models.BPE) and tokenizer.model.unk_token is None:
        tokenizer.model.unk_token = "[UNK]"
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    if tokenizer.post_processor is None:
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [EOS] [BOS] $B [EOS]",
            special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)]
        )
    return tokenizer

tokenizer = train_or_load_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print(f"📊 Vocab size: {vocab_size}")

BOS_ID = tokenizer.token_to_id("[BOS]")
EOS_ID = tokenizer.token_to_id("[EOS]")

def encode_text_stream_with_boundaries(txt: str) -> list[int]:
    ids: list[int] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        ids.extend(tokenizer.encode(line).ids)  # post-processor injects BOS/EOS
    return ids

def decode_ids(ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)

# ---------------------
# Load dataset
# ---------------------
with open(cfg.dataset_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
print(f"📖 Dataset length (chars): {len(raw_text):,}")

all_ids = encode_text_stream_with_boundaries(raw_text)
data = torch.tensor(all_ids, dtype=torch.long)
if len(data) < cfg.block_size + 1:
    raise ValueError(f"Encoded dataset too small ({len(data)}) for block_size={cfg.block_size}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str):
    src = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(src) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([src[i:i+cfg.block_size] for i in ix])
    y = torch.stack([src[i+1:i+cfg.block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ---------------------
# LR schedule
# ---------------------
def cosine_lr(step: int, base_lr: float, min_lr: float, warmup: int, total: int):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

# ---------------------
# Model
# ---------------------
class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        # AMP-stable mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, torch.finfo(wei.dtype).min)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # GELU generally improves GPTs
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop_emb = nn.Dropout(cfg.emb_dropout)  # new: embedding dropout
        self.blocks = nn.Sequential(*[Block(cfg.n_embd, cfg.n_head) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.token_embedding_table.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.drop_emb(tok_emb + pos_emb)  # apply embedding dropout
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B*T, -1),
                targets.view(B*T),
                label_smoothing=cfg.label_smoothing,  # new: label smoothing
            )
        return logits, loss

    @torch.no_grad()
    def _sample_next(self, logits_last, temperature=1.0, top_k=50):
        if temperature != 1.0:
            logits_last = logits_last / temperature
        probs = F.softmax(logits_last, dim=-1)
        if top_k is not None and top_k > 0:
            v, ix = torch.topk(probs, top_k)
            mask = torch.ones_like(probs, dtype=torch.bool)
            mask.scatter_(1, ix, False)
            probs = probs.masked_fill(mask, 0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50, eos_id: int | None = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -cfg.block_size:]
            logits, _ = self(idx_cond)
            next_token = self._sample_next(logits[:, -1, :], temperature, top_k)
            idx = torch.cat((idx, next_token), dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break
        return idx

# ---------------------
# Init model/opt/amp
# ---------------------
model = GPTLanguageModel().to(device)

# AdamW param groups: no weight decay on LayerNorm or bias
decay = []
no_decay = []
for name, param in model.named_parameters():
    if name.endswith('bias') or 'ln' in name or 'norm' in name:
        no_decay.append(param)
    else:
        decay.append(param)

optim_groups = [
    {"params": decay, "weight_decay": cfg.weight_decay},
    {"params": no_decay, "weight_decay": 0.0},
]

optimizer = torch.optim.AdamW(
    optim_groups,
    lr=cfg.learning_rate,
    betas=cfg.betas
)

# Use new torch.amp API (fixes deprecation warnings)
scaler = torch.amp.GradScaler('cuda', enabled=(cfg.use_amp and device == 'cuda'))

param_millions = sum(p.numel() for p in model.parameters())/1e6
print(f"🤖 Model parameters: {param_millions:.2f} M")

os.makedirs(cfg.ckpt_dir, exist_ok=True)

start_iter = 0
best_val_loss = float('inf')
no_improve_evals = 0
if cfg.resume_path is not None and os.path.exists(cfg.resume_path):
    ckpt = torch.load(cfg.resume_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['opt'])
    start_iter = ckpt.get('iter', 0) + 1
    best_val_loss = ckpt.get('val_loss', float('inf'))
    print(f"🔄 Resumed from {cfg.resume_path} at iter {start_iter}, best_val_loss={best_val_loss:.4f}")

@torch.no_grad()
def estimate_metrics():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_iters, device=device)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast('cuda', enabled=(cfg.use_amp and device == 'cuda')):
                _, loss = model(X, Y)
            losses[k] = loss
        mean_loss = losses.mean().item()
        out[split] = {'loss': mean_loss, 'ppl': math.exp(mean_loss)}
    model.train()
    return out

# ---------------------
# Training
# ---------------------
t0 = time.time()
for iter in range(start_iter, cfg.max_iters):
    lr = cosine_lr(iter, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters, cfg.max_iters)
    for g in optimizer.param_groups:
        g['lr'] = lr

    if iter % 100 == 0 and iter > 0:
        elapsed = time.time() - t0
        print(f"[Progress: {iter}/{cfg.max_iters} ({100*iter/cfg.max_iters:.1f}%)] elapsed {elapsed/60:.1f} min")

    if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
        metrics = estimate_metrics()
        print(f"Step {iter}: train loss {metrics['train']['loss']:.4f} (ppl {metrics['train']['ppl']:.2f}), "
              f"val loss {metrics['val']['loss']:.4f} (ppl {metrics['val']['ppl']:.2f}), lr {lr:.2e}")
        if metrics['val']['loss'] + 1e-6 < best_val_loss:
            best_val_loss = metrics['val']['loss']
            no_improve_evals = 0
            best_path = os.path.join(cfg.ckpt_dir, "best.pt")
            torch.save({'iter': iter, 'model': model.state_dict(), 'opt': optimizer.state_dict(),
                        'val_loss': best_val_loss, 'cfg': cfg.__dict__}, best_path)
            print(f" -> 💾 Best checkpoint updated: {best_path}")
        else:
            no_improve_evals += 1
            if no_improve_evals >= cfg.patience_evals:
                print(f"⏹️ Early stopping after {no_improve_evals} evals without val improvement.")
                break

    optimizer.zero_grad(set_to_none=True)
    for micro in range(cfg.grad_accum_steps):
        xb, yb = get_batch('train')
        with torch.amp.autocast('cuda', enabled=(cfg.use_amp and device == 'cuda')):
            _, loss = model(xb, yb)
            loss = loss / cfg.grad_accum_steps
        scaler.scale(loss).backward()

    if cfg.grad_clip and cfg.grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

    scaler.step(optimizer)
    scaler.update()

    if iter > 0 and iter % cfg.save_every == 0:
        ckpt_path = os.path.join(cfg.ckpt_dir, f"checkpoint_{iter}.pt")
        torch.save({'iter': iter, 'model': model.state_dict(), 'opt': optimizer.state_dict(),
                    'val_loss': best_val_loss, 'cfg': cfg.__dict__}, ckpt_path)
        print(f"💾 Checkpoint saved at step {iter}: {ckpt_path}")

# ---------------------
# Save final, generate sample
# ---------------------
final_model_path = os.path.join(cfg.ckpt_dir, "gpt_final.pt")
torch.save(model.state_dict(), final_model_path)
print(f"✅ Final model saved to {final_model_path}")

context = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
generated_ids = model.generate(
    context,
    max_new_tokens=cfg.gen_max_new_tokens,
    temperature=cfg.gen_temperature,
    top_k=cfg.gen_top_k,
    eos_id=EOS_ID
)[0].tolist()
print("📝 Sample:\n", decode_ids(generated_ids))

# ---------------------
# Zip checkpoints for download (COLAB)
# ---------------------
print("\n📦 Zipping checkpoints for download...")
os.system(f"zip -r checkpoints.zip {cfg.ckpt_dir}")
print("✅ Run this in a new cell to download:")
print("   from google.colab import files")
print("   files.download('checkpoints.zip')")