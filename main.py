import torch
from BigramLanguageModel import BigramLanguageModel

from data import Data
from hyperparams import eval_iters, device, learning_rate, max_iters, eval_interval, block_size, batch_size

torch.manual_seed(6942)


@torch.no_grad()
def est_loss():
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


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


if __name__ == "__main__":
    context = torch.zeros((5, 5), dtype=torch.long, device=device)

    data_holder = Data('input.txt')
    data = data_holder.tensor
    print(data_holder.encode("abcdefg"))

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = BigramLanguageModel()
    m = model.to(device)

    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = est_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model

    sentences = [data_holder.encode("do you not hear me speak?")]
    sentenceTensor = torch.tensor(sentences)
    text = m.generate(sentenceTensor, max_new_tokens=100)[0].tolist()
    print(data_holder.decode(text))
