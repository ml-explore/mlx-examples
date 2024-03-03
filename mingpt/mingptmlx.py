# %% [markdown]
# # MLX MinGPT
# 

# %%
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

# %%

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.01


# %%
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = mx.default_device()


# %%
# ------------

mx.random.seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# %%

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# %%

# Train and test splits
data = mx.array(encode(text), dtype=mx.int64)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# %%

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = mx.random.randint(0, len(data) - block_size, (batch_size,))
    ix = [i.item() for i in ix]
    x = mx.stack([data[i:i+block_size] for i in ix])
    y = mx.stack([data[i+1:i+block_size+1] for i in ix])
    # x, y = x.to(device), y.to(device)
    return x, y


# %%

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril = mx.tril(mx.ones([block_size, block_size]))
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose((0,2,1)) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        mask = self.tril[:T, :T] == 0
        wei = mx.where(mask, float('-inf'), wei) # (B, T, T)
        wei = nn.softmax(wei, axis=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


# %%

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)
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

    def __call__(self, x):
        return self.net(x)


# %%

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

    def __call__(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# %%
# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(mx.arange(T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = nn.losses.cross_entropy(logits, targets, reduction='mean')

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = nn.softmax(logits, axis=-1) # (B, C)
            # sample from the distribution
            idx_next = mx.random.categorical(probs, axis=-1, num_samples=1)
            # append sampled index to the running sequence
            idx = mx.concatenate((idx, idx_next), axis=1) # (B, T+1)
        return idx


# %%

model = BigramLanguageModel()

# %%
params = model.parameters()


# %%
# print the number of parameters in the model
p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
print(f"Total parameters {p:.3f}M")

# %%
# create an optimizer
optimizer = optim.AdamW(learning_rate)

# %%
def estimate_loss(model):
    out = {}
    for split in ['train', 'val']:
        losses = mx.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def loss_fn(model, x, y):
    _, loss = model(x, y)
    return loss

def step(model, optimizer, inputs, targets):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, inputs, targets)
    optimizer.update(model, grads)
    return loss


# %%
model.train()
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train'].item():.4f}, val loss {losses['val'].item():.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    step(model, optimizer, xb, yb)

print("Training complete")

# %%
# save weights
model.save_weights("./shakespearebigramweights.npz")

# %%
def generate(model, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits, _ = model(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = nn.softmax(logits, axis=-1) # (B, C)
        # sample from the distribution
        idx_next = mx.random.categorical(probs, axis=-1, num_samples=10)
        # append sampled index to the running sequence
        idx = mx.concatenate((idx, idx_next), axis=1) # (B, T+1)
    return idx


# %%
# generate from the model
model.eval()
print("Generating..")
context = mx.zeros((1, 1), dtype=mx.int64)
# generated = model.generate(context, max_new_tokens=1000)[0].tolist()
generated = generate(model, context, max_new_tokens=1000)[0].tolist()
print(decode(generated))



# %%
load_weights = True
model = BigramLanguageModel()
if load_weights:
    model.load_weights("./shakespearebigramweights.npz")
context = mx.zeros((1, 1), dtype=mx.int64)
generated = model.generate(context, max_new_tokens=1000)[0].tolist()
print(decode(generated))


