import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cpu'  # Selected when training or loading a checkpoint.
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

# Importing this module does not load data or start training.
from pathlib import Path
import argparse
import os

vocab_size = 0
chars = []


def select_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def prepare_data(path):
    global chars, vocab_size, train_data, val_data
    text = Path(path).read_text(encoding='utf-8')
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    if min(len(train_data), len(val_data)) <= block_size:
        raise ValueError('Both data splits must be longer than the context window.')


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Decoder-only transformer; class name retained from the tutorial.
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


CONFIG_KEYS = ('block_size', 'n_embd', 'n_head', 'n_layer', 'dropout', 'vocab_size')


def save_checkpoint(path, model, optimizer, step, losses=None):
    """Save atomically so an interrupted write does not replace a good checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'format_version': 1,
        'model_config': {key: globals()[key] for key in CONFIG_KEYS},
        'chars': chars,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'losses': losses,
    }
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, requested_device=None):
    """Restore architecture and vocabulary without needing the training text."""
    global device, chars
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format.')
    for key in CONFIG_KEYS:
        globals()[key] = checkpoint['model_config'][key]
    chars = checkpoint['chars']
    if len(chars) != vocab_size:
        raise ValueError('Checkpoint vocabulary does not match the model.')
    device = select_device(requested_device)
    model = BigramLanguageModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, chars, checkpoint['step']


def train(args):
    global device, batch_size, eval_iters
    if min(args.steps, args.eval_interval, args.eval_iters, args.batch_size) < 1:
        raise ValueError('Steps, evaluation interval, evaluation batches and batch size must be positive.')
    torch.manual_seed(1337)
    device = select_device(args.device)
    batch_size, eval_iters = args.batch_size, args.eval_iters
    prepare_data(args.input)
    model = BigramLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    checkpoint_dir = Path(args.checkpoint_dir)
    best_val = float('inf')
    completed = 0
    print(f'Using {device}; checkpoints: {checkpoint_dir.resolve()}', flush=True)
    try:
        for iteration in range(args.steps):
            xb, yb = get_batch('train')
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            completed = iteration + 1
            if completed % args.eval_interval == 0 or completed == args.steps:
                losses = {key: float(value) for key, value in estimate_loss(model).items()}
                print(f"step {completed}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
                save_checkpoint(checkpoint_dir / 'latest.pt', model, optimizer, completed, losses)
                if losses['val'] < best_val:
                    best_val = losses['val']
                    save_checkpoint(checkpoint_dir / 'best.pt', model, optimizer, completed, losses)
        save_checkpoint(checkpoint_dir / 'final.pt', model, optimizer, completed, losses)
    except KeyboardInterrupt:
        save_checkpoint(checkpoint_dir / 'interrupted.pt', model, optimizer, completed)
        print(f'Interrupted; saved current weights after {completed} completed steps.', flush=True)
        return
    print('Training finished. Use generate.py to load a checkpoint and generate text.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the character-level transformer and save checkpoints.')
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('input.txt'))
    parser.add_argument('--steps', type=int, default=max_iters)
    parser.add_argument('--eval-interval', type=int, default=eval_interval)
    parser.add_argument('--eval-iters', type=int, default=eval_iters)
    parser.add_argument('--batch-size', type=int, default=batch_size)
    parser.add_argument('--checkpoint-dir', type=Path, default=Path(__file__).parent / 'checkpoints')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    train(parser.parse_args())
