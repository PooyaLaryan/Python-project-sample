import torch
from torch import nn
import matplotlib.pyplot as plt

channels = 3 
h = w = 5
x = torch.rand(channels, w, h)  

# img = x.permute(1, 2, 0).numpy()

# plt.imshow(img)
# plt.title("Random 5x5 RGB Image")
# plt.axis("off")
# plt.show()

print(x)
print(x.shape)

# Formula 
# 1 + ((floow( input size + 2 × padding - kernel size)))/stride ))

pool = nn.MaxPool2d(kernel_size=2, stride=2)
out = pool(x.unsqueeze(0))  # اضافه کردن batch dimension
print(out)
print(out.shape)

pool = nn.MaxPool2d(kernel_size=2, stride=1)
out = pool(x.unsqueeze(0))  # اضافه کردن batch dimension
print(out)
print(out.shape)

pool = nn.MaxPool2d(kernel_size=3, stride=1)
out = pool(x.unsqueeze(0))  # اضافه کردن batch dimension
print(out)
print(out.shape)