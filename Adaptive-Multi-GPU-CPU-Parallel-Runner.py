import torch
import torch.multiprocessing as mp
import concurrent.futures
import os
import time


def heavy_gpu_task(rank, return_dict):
    """تسک سنگین روی GPU"""
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    print(f"[GPU {rank}] Task started on {device}")
    start_time = time.time()

    x = torch.randn(6000, 6000, device=device)
    y = torch.randn(6000, 6000, device=device)
    z = torch.matmul(x, y)
    torch.cuda.synchronize()

    duration = time.time() - start_time
    result_value = z.mean().item()

    print(f"[GPU {rank}] ✅ Task finished in {duration:.2f} sec")
    return_dict[rank] = (result_value, duration)


def heavy_cpu_task(index):
    """تسک سنگین روی CPU"""
    print(f"[CPU {index}] Task started...")
    start_time = time.time()

    x = torch.randn(4000, 4000)
    y = torch.randn(4000, 4000)
    z = torch.matmul(x, y)

    duration = time.time() - start_time
    result_value = z.mean().item()

    print(f"[CPU {index}] ✅ Task finished in {duration:.2f} sec")
    return (index, result_value, duration)


def main():
    os.environ["OMP_NUM_THREADS"] = "1"  # جلوگیری از oversubscription

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🚀 {gpu_count} GPU(s) detected.")

        if gpu_count > 1:
            # اجرای هم‌زمان روی چند GPU
            print("🔧 Running tasks in parallel on multiple GPUs...")
            manager = mp.Manager()
            return_dict = manager.dict()
            processes = []

            for rank in range(gpu_count):
                p = mp.Process(target=heavy_gpu_task, args=(rank, return_dict))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            print("\n🎯 All GPU tasks completed.")
            for gpu_id, (result, duration) in return_dict.items():
                print(f"  🟩 GPU {gpu_id}: mean={result:.5f}, time={duration:.2f} sec")

        else:
            # اجرای پشت سر هم روی یک GPU
            print("⚙️ Only one GPU found. Running tasks sequentially on GPU 0...")
            device = "cuda:0"
            for i in range(2):
                start = time.time()
                x = torch.randn(5000, 5000, device=device)
                y = torch.randn(5000, 5000, device=device)
                z = torch.matmul(x, y)
                torch.cuda.synchronize()
                print(f"  ✅ Task {i+1} finished (mean={z.mean().item():.4f}) in {time.time() - start:.2f}s")

    else:
        # اجرای موازی روی CPU
        print("⚙️ No GPU found. Running tasks in parallel on CPU cores...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(heavy_cpu_task, i) for i in range(4)]
            for f in concurrent.futures.as_completed(futures):
                index, result, duration = f.result()
                print(f"  🟨 CPU Task {index}: mean={result:.5f}, time={duration:.2f}s")

    print("\n✅ All tasks done. Continuing the rest of the program...")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
