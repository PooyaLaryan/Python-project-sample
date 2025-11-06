import torch
import torch.nn as nn

# تنظیمات شبکه
seq_len = 3
input_size = 4
hidden_size = 10
num_layers = 3
batch_size = 1

# شبکه LSTM
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
fc = nn.Linear(hidden_size, 1)

# ورودی و هدف
x = torch.tensor([[[0.1,0.2,0.3,0.4],
                   [0.5,0.6,0.7,0.8],
                   [0.9,1.0,1.1,1.2]]], dtype=torch.float32)
y = torch.tensor([[0.5]], dtype=torch.float32)

# بهینه‌ساز
optimizer = torch.optim.SGD(list(lstm.parameters()) + list(fc.parameters()), lr=0.1)
criterion = nn.MSELoss()

# یک epoch ساده
optimizer.zero_grad()
out, (h_n, c_n) = lstm(x)
pred = fc(out[:, -1, :])
loss = criterion(pred, y)
loss.backward()

print("=== Before update ===")
for name, param in lstm.named_parameters():
    print(name, param.data.flatten()[:5])  # نمایش چند وزن اول

optimizer.step()

print("\n=== After update ===")
for name, param in lstm.named_parameters():
    print(name, param.data.flatten()[:5])