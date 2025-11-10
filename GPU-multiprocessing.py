import torch
import torch.multiprocessing as mp
import os
import time

def heavy_gpu_task(rank, return_dict):
    """
    rank: شماره‌ی GPU که باید روی آن اجرا شود
    return_dict: دیکشنری اشتراکی برای ذخیره نتیجه هر GPU
    """
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    print(f"[GPU {rank}] Task started on {device}")

    torch.cuda.set_device(rank)
    start_time = time.time()

    # کار سنگین — ضرب دو ماتریس بزرگ
    x = torch.randn(6000, 6000, device=device)
    y = torch.randn(6000, 6000, device=device)
    z = torch.matmul(x, y)
    torch.cuda.synchronize()

    duration = time.time() - start_time
    result_value = z.mean().item()

    print(f"[GPU {rank}] ✅ Task finished in {duration:.2f} sec")
    return_dict[rank] = (result_value, duration)

def main():
    if not torch.cuda.is_available():
        print("❌ No GPU found. Please run on a system with CUDA support.")
        return

    gpu_count = torch.cuda.device_count()
    print(f"🚀 Found {gpu_count} GPU(s). Launching parallel processes...")

    # دیکشنری اشتراکی برای دریافت خروجی‌ها
    manager = mp.Manager()
    return_dict = manager.dict()

    # ایجاد یک Process برای هر GPU
    processes = []
    for rank in range(gpu_count):
        p = mp.Process(target=heavy_gpu_task, args=(rank, return_dict))
        p.start()
        processes.append(p)

    # منتظر ماندن برای اتمام همه فرآیندها
    for p in processes:
        p.join()

    print("\n🎯 All GPU tasks completed.")
    for gpu_id, (result, duration) in return_dict.items():
        print(f"  🟩 GPU {gpu_id}: mean={result:.5f}, time={duration:.2f} sec")

    print("\n➡ Continuing the rest of the program...")

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"  # جلوگیری از oversubscription
    mp.set_start_method("spawn", force=True)
    main()
