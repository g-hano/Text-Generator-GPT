import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions
max_iters  = 1000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}\n")
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
# -----------------

with open('RomeoJuliet.txt', encoding="utf-8") as f:
    text = f.read()


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)} #? # yukarda chars'ı yazdırdık, boşluk 1 numara, ünlem 2 numara.
itos = {i:ch for i,ch in enumerate(chars)} #? # bu da her karaktere rakam ataması yapıyo, ama biz bu sefer rakam-harf olarak kaydediyoz
                                           

encode = lambda s: [stoi[c] for c in s] #? # enumerate chars'ın içinde tek tek gidip her 
                                        #? karaktere rakam ataması yapıyo, harf-rakam olarak kaydediyo

decode = lambda l: "".join([itos[i] for i in l]) #? boşluk eklemek önemli, boşluk, yazılar için önemli

# train and validation splits
data = torch.tensor(encode(text), dtype=torch.long) #? yazının içindeki her karaktere sayı verip, tensore çeviriyo

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split.lower() == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, )) #? 0 ile data-8 sayısı arasından, rastgele 4 tane sayı getirir,her sayı 1 yönlüdür dimension
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
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
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B,T,C)
        q = self.query(x)   # (B,T,C)
        # compute attention scores ["affinities"]
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ muliple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        super().__init__()
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C) -> (Batch, Time, Channel)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #* F.cross_entropy wants logits like that
            targets = targets.view(B*T) #* F.cross_entropy wants targets in one dim
            loss = F.cross_entropy(logits, targets)

        """
        The F.cross_entropy function in PyTorch
        expects the logits tensor to have shape (N, C)
        and the targets tensor to have shape (N,),
        where N is the number of examples (batch size)
        and C is the number of classes (in this case, vocab_size).
        This function computes the cross-entropy loss
        between the predicted logits and the target labels.
        """        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond) #! idx'i kendine sokup, logits ve loss'u alıyo. 
                                     #! Kendine sokabiliyo çünkü bu BigramLanguageModel sınıfı bir değişken
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C), 
                                      #! logits normalde (B,T,C), biz B'nin hepsini, T'nin sonuncusunu, C'nin hepsini istiyoruz. 
                                      #! çünkü T, bir sonraki karakterle ilgili ip ucu veriyo bize, bir sonraki karaktere baglı
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) #! numsamples make 1 in (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            """
            fonksiyon idx istiyo,
            aldigi idx'i kendine sokup logits ve loss veriyo
            logits'in normalde şekli (B, T, C),
            ama bu şekildeyken işe yaramaz
            o yüzden B ve C'nin tamamini, T'nin sonuncusunu aliyoz.

            probs'a gelince, B ve C'yi ona veriyoz, ihtimal aliyoz,
            idx_next' e veriyoz, 1 tane fazladan aliyoz,bu fazla olan bir sonraki gelecek harf.

            idx'i, idx_next'le birleştirioz. artik bir sonraki harf idx in içinde
            en son idx'i döndürüyoz
            """
        return idx
        
model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    torch.autograd.set_detect_anomaly(True)
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))