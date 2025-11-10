import time
import concurrent.futures

def task_one():
    print("🔹 Task 1 started...")
    time.sleep(2)  # شبیه‌سازی کار زمان‌بر (CPU یا I/O)
    print("✅ Task 1 finished.")
    return "Result from Task 1"

def task_two():
    print("🔹 Task 2 started...")
    time.sleep(3)
    print("✅ Task 2 finished.")
    return "Result from Task 2"

def main():
    print("🚀 Starting both tasks in separate threads...")

    # ایجاد ThreadPool برای اجرای موازی
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # اجرای دو تابع به صورت هم‌زمان
        future1 = executor.submit(task_one)
        future2 = executor.submit(task_two)

        # منتظر ماندن تا هر دو تمام شوند
        result1 = future1.result()
        result2 = future2.result()

    print("🎯 Both tasks completed!")
    print("Task1:", result1)
    print("Task2:", result2)
    print("➡ Continuing the rest of the program...")

if __name__ == "__main__":
    main()
