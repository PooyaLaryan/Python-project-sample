import numpy as np

# ============================================
# 1. فرض کن یک Feature Map ساده داریم
# ============================================

# شکل: (Channel, Height, Width)
# یعنی 3 کانال و هر کانال 3x3
feature_map = np.array([
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],

    [[9, 8, 7],
     [6, 5, 4],
     [3, 2, 1]],

    [[2, 2, 2],
     [3, 3, 3],
     [4, 4, 4]]
])

print("📘 Feature map shape:", feature_map.shape)
print("Feature map (C=3, H=3, W=3):\n", feature_map)
print("---------------------------------------------------")

# ============================================
# 2. محاسبه‌ی average و max در بعد channel
# ============================================

avg_map = np.mean(feature_map, axis=0)
max_map = np.max(feature_map, axis=0)

print("🔹 Average Pool across channels:\n", avg_map)
print("🔹 Max Pool across channels:\n", max_map)
print("---------------------------------------------------")

# ============================================
# 3. ترکیب avg و max → concat در بعد channel
# ============================================

concat = np.stack([avg_map, max_map], axis=0)  # (2, 3, 3)
print("🔹 Concatenated (2 channels: avg & max):\n", concat)
print("---------------------------------------------------")

# ============================================
# 4. اعمال فیلتر 3x3 ساده برای تولید Attention Map
# ============================================

# تعریف فیلتر ساده (می‌تونی خودت مقدار دهی کنی)
kernel = np.ones((3, 3)) / 9.0  # میانگین‌گیر
print("Kernel (3x3):\n", kernel)

# انجام کانولوشن ساده روی هر کانال
# (اینجا فقط نمونه‌ی دستی با padding=1)
def conv2d_simple(x, kernel):
    H, W = x.shape
    kH, kW = kernel.shape
    pad = 1
    x_pad = np.pad(x, pad, mode='constant', constant_values=0)
    y = np.zeros_like(x)
    for i in range(H):
        for j in range(W):
            region = x_pad[i:i+kH, j:j+kW]
            y[i, j] = np.sum(region * kernel)
    return y

conv_avg = conv2d_simple(concat[0], kernel)
conv_max = conv2d_simple(concat[1], kernel)
conv_sum = conv_avg + conv_max

print("🔹 Convolution result (avg branch):\n", conv_avg)
print("🔹 Convolution result (max branch):\n", conv_max)
print("🔹 Combined result:\n", conv_sum)
print("---------------------------------------------------")

# ============================================
# 5. اعمال تابع sigmoid برای ساخت attention map
# ============================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

attention_map = sigmoid(conv_sum)
print("🔥 Attention Map (after sigmoid):\n", np.round(attention_map, 3))
print("---------------------------------------------------")

# ============================================
# 6. اعمال Attention روی feature map اولیه
# ============================================

# ضرب هر کانال در attention_map
output = feature_map * attention_map
print("✅ Output feature maps after applying attention:\n", np.round(output, 2))
