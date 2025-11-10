import time
import concurrent.futures
import torch  # فقط اگر کار GPU داری

def heavy_task_one():
    print("🔹 Task 1 (GPU/CPU) started...")
    time.sleep(1)  # شبیه‌سازی آماده‌سازی اولیه

    # مثال: عملیات ماتریسی روی GPU (اگر موجود باشد)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Task 1 running on: {device}")

    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)
    z = torch.matmul(x, y)  # ضرب ماتریسی سنگین
    torch.cuda.synchronize() if device == "cuda" else None

    print("✅ Task 1 finished.")
    return f"Result 1 (sum={z.sum().item():.4f})"

def heavy_task_two():
    print("🔹 Task 2 (GPU/CPU) started...")
    time.sleep(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Task 2 running on: {device}")

    x = torch.randn(4000, 4000, device=device)
    y = torch.randn(4000, 4000, device=device)
    z = torch.matmul(x, y)
    torch.cuda.synchronize() if device == "cuda" else None

    print("✅ Task 2 finished.")
    return f"Result 2 (mean={z.mean().item():.4f})"

def main():
    print("🚀 Starting heavy GPU/CPU tasks in parallel...")

    # اجرای دو پردازش مستقل
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future1 = executor.submit(heavy_task_one)
        future2 = executor.submit(heavy_task_two)

        # منتظر می‌مانیم تا هر دو پردازش تمام شوند
        result1 = future1.result()
        result2 = future2.result()

    print("\n🎯 Both tasks completed!")
    print(result1)
    print(result2)
    print("➡ Continuing the rest of the program...")

if __name__ == "__main__":
    main()
