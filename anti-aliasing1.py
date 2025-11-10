import cv2
import numpy as np
import os

# ---------- تنظیمات ----------
canvas_size = 512   # اندازه تصویر اصلی
circle_radius = 200  # شعاع دایره
downscale_size = 128 # اندازه هدف
ssaa_factor = 3      # فاکتور supersampling

# ---------- ایجاد تصویر دایره ----------
def make_circle_image(size, radius):
    img = np.ones((size, size, 3), dtype=np.uint8) * 255  # سفید
    center = (size // 2, size // 2)
    cv2.circle(img, center, radius, (0, 0, 0), thickness=-1, lineType=cv2.LINE_8)
    return img

# ---------- روش‌های مختلف آنتی‌الیاس ----------
def resize_nearest(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)

def resize_bilinear(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

def resize_bicubic(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

def supersample_ssaa(img, target_size, factor=3):
    up = cv2.resize(img, (target_size*factor, target_size*factor), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(up, (5,5), sigmaX=1.2)
    return cv2.resize(blurred, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

def prefilter_gaussian(img, target_size):
    blur = cv2.GaussianBlur(img, (7,7), sigmaX=1.5)
    return cv2.resize(blur, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

def fxaa_like(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    mask = (mag > 0.15).astype(np.float32)
    blurred = cv2.GaussianBlur(img, (5,5), sigmaX=0.5)
    mask3 = np.stack([mask]*3, axis=-1)
    out = (img*(1-mask3) + blurred*mask3).astype(np.uint8)
    return out

# ---------- اجرای آزمایش ----------
img = make_circle_image(canvas_size, circle_radius)

results = {
    "original": img,
    "nearest": resize_nearest(img, downscale_size),
    "bilinear": resize_bilinear(img, downscale_size),
    "bicubic": resize_bicubic(img, downscale_size),
    "ssaa": supersample_ssaa(img, downscale_size, ssaa_factor),
    "gaussian_prefilter": prefilter_gaussian(img, downscale_size),
}

# FXAA-like روی نسخه bilinear
results["fxaa_like"] = fxaa_like(results["bilinear"])

# ---------- نمایش و ذخیره ----------
os.makedirs("aa_demo_results", exist_ok=True)

for name, im in results.items():
    cv2.imwrite(f"aa_demo_results/{name}.png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    print(f"Saved: aa_demo_results/{name}.png")

# ساخت تصویر مقایسه‌ای کنار هم
strip = np.hstack([
    results["nearest"],
    results["bilinear"],
    results["bicubic"],
    results["ssaa"],
    results["gaussian_prefilter"],
    results["fxaa_like"]
])

cv2.imwrite("aa_demo_results/comparison_strip.png", cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
print("✅ Saved comparison_strip.png in aa_demo_results folder")
