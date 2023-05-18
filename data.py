import torch


class Data:

    def __init__(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            self.read_file = f.read()
            self.self_text = self.read_file.lower()

        self.chars = sorted(list(set(self.self_text)))
        self.vocab_size = len(self.chars)
        print(self.vocab_size)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.tensor = torch.tensor(self.encode(self.self_text), dtype=torch.long)

    def encode(self, t):
        return [self.stoi[c] for c in t]

    def decode(self, t):
        return ''.join([self.itos[i] for i in t])
