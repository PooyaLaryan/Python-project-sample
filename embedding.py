import torch
from torch import nn
import random

data_list = random.sample(range(1, 30), 10)

print(data_list)

data = torch.tensor(data_list, dtype=torch.long)

vocab_size = data.max().item() + 1

print(data.max())
print(data.max().item())
print(vocab_size)
print(len(data_list))
print(len(data))
embedding = nn.Embedding(vocab_size,16, padding_idx=0)
data_embedding = embedding(data)
print(data_embedding)
print(data_embedding.shape)
