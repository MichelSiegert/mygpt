import torch

batch_size = 64
block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_head = 4
n_layer = 4
n_embd = 64
vocab_size = 39
dropout = 0.0

max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200