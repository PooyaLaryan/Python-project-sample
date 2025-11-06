import torch
import torch.nn as nn


# Formula 
# 1 + ((floow( input size + 2 × padding - kernel size)))/stride ))

x = torch.rand(1, 3, 10, 10)  # batch=1, channels=3, 10x10 image

conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
y = conv(x)

print(y.shape)



x = torch.rand(1, 3, 10, 7)  # batch=1, channels=3, 10x7 image
conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
y = conv(x)

print(y.shape) # torch.Size([1, 6, 8, 5])


x = torch.rand(1, 3, 10, 10)

conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0)
pool = nn.MaxPool2d(kernel_size=2, stride=2)
conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=0)

y1 = conv1(x)
y2 = pool(y1)
y3 = conv2(y2)

print("After Conv1:", y1.shape)  # (1, 6, 8, 8)
print("After Pool:", y2.shape)   # (1, 6, 4, 4)
print("After Conv2:", y3.shape)  # (1, 12, 2, 2)




x = torch.rand(1, 3, 10, 10)

conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0)
y = conv1(x)            # (1,6,8,8)

pool = nn.MaxPool2d(2,2)
avg = nn.AvgPool2d(2,2)

y_max = pool(y)         # (1,6,4,4)
y_avg = avg(y)          # (1,6,4,4)

concat = torch.cat([y_max, y_avg], dim=1)  # (1,12,4,4)

# مثال: Conv بعدی که spatial را نگه می‌دارد و کانال خروجی را 24 می‌کند
conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
out = conv2(concat)    # (1,24,4,4)

print(concat.shape, out.shape)