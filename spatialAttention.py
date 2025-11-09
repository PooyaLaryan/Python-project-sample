import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ======================
# 1. ساخت تصویر مصنوعی
# ======================
W = H = 128
img = Image.new("RGB", (W, H), (30, 30, 30))
draw = ImageDraw.Draw(img)
draw.rectangle([20, 20, 90, 70], fill=(200, 40, 40))   # مستطیل قرمز
draw.ellipse([60, 60, 110, 110], fill=(40, 200, 40))   # دایره سبز
draw.line([0, 127, 127, 0], fill=(40, 40, 200), width=3)  # خط آبی

img_np = np.array(img).astype(np.float32) / 255.0
img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

# ======================
# 2. شبکه‌ی CNN کوچک
# ======================
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

net = SmallCNN()
with torch.no_grad():
    feat = net(img_t)  # (1,C,Hf,Wf)

# تابع کمکی برای نمایش تصویر
def show_image(arr, title=None):
    plt.figure(figsize=(4,4))
    if arr.ndim == 2:
        plt.imshow(arr, cmap='viridis')
    else:
        plt.imshow(np.clip(arr, 0, 1))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

# ======================
# 3. نمایش مراحل Spatial Attention
# ======================

# 1) تصویر اصلی
show_image(img_np, "Original Image")

# 2) چند ویژگی اولیه از خروجی CNN
feat_np = feat.squeeze(0).numpy()
for i in range(min(3, feat_np.shape[0])):
    plt.figure(figsize=(4,4))
    plt.imshow(feat_np[i], cmap='magma')
    plt.title(f"Feature Map {i}")
    plt.axis('off')
    plt.show()

# 3) محاسبه‌ی average و max pool در بعد channel
avg_pool = torch.mean(feat, dim=1, keepdim=True)
max_pool, _ = torch.max(feat, dim=1, keepdim=True)

show_image(avg_pool.squeeze().numpy(), "Channel-wise Average Pool")
show_image(max_pool.squeeze().numpy(), "Channel-wise Max Pool")

# 4) ترکیب avg و max و عبور از conv7x7 و sigmoid
concat = torch.cat([avg_pool, max_pool], dim=1)
conv7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
with torch.no_grad():
    att_logits = conv7(concat)
att_map = torch.sigmoid(att_logits)

show_image(att_map.squeeze().numpy(), "Spatial Attention Map")

# 5) اعمال attention روی feature maps
att_applied = feat * att_map
sum_before = feat_np.sum(axis=0)
sum_after = att_applied.squeeze(0).numpy().sum(axis=0)

show_image(sum_before, "Sum of Features (Before Attention)")
show_image(sum_after, "Sum of Features (After Attention)")

# 6) ترکیب attention با تصویر اصلی به صورت Heatmap
att_np = att_map.squeeze().cpu().numpy()
att_min, att_max = float(att_np.min()), float(att_np.max())
att_norm = (att_np - att_min) / (att_max - att_min + 1e-9)

# 🔹 resize attention map از 32x32 به 128x128
from PIL import Image
att_resized = np.array(
    Image.fromarray(att_norm).resize((W, H), resample=Image.BILINEAR)
)

# 🔹 تبدیل به رنگی (heatmap)
cmap = plt.get_cmap('viridis')
att_color = cmap(att_resized)[:, :, :3]

# 🔹 ترکیب با تصویر اصلی
overlay = 0.6 * img_np + 0.4 * att_color
show_image(overlay, "Attention Heatmap Overlay")