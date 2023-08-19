import torch
from torch.utils.data import Dataset
from pathlib import Path


class Shakespear(Dataset):
    def __init__(self, path: Path, block_size=8):
        self.path = path
        self.block_size = block_size
        self.text = self._get_text()
        self.data = self.encode(self.text)
        self._make_data()

    def _get_text(self):
        with open(self.path) as f:
            text = f.read()
            self.chars = sorted(list(set(text)))
            self.vocab_size = len(self.chars)

            return text

    def _make_data(self) -> None:
        ix = range(len(self.data) - self.block_size)
        self.x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        self.y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])

    def encode(self, string):
        stoi = {ch: i for i, ch in enumerate(self.chars)}

        return torch.tensor([stoi[c] for c in string])

    def decode(self, tokens):
        itos = {i: ch for i, ch in enumerate(self.chars)}

        return ''.join([itos[t] for t in tokens])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
