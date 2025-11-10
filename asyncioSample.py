import asyncio

# تابع اول
async def task_one():
    print("🔹 Task 1 started...")
    await asyncio.sleep(10)  # شبیه‌سازی کار زمان‌بر
    print("✅ Task 1 finished.")
    return "Result from Task 1"

# تابع دوم
async def task_two():
    print("🔹 Task 2 started...")
    await asyncio.sleep(6)  # شبیه‌سازی کار زمان‌بر
    print("✅ Task 2 finished.")
    return "Result from Task 2"

# تابع اصلی
async def main():
    print("🚀 Starting both tasks...")
    
    # هم‌زمان اجرا کردن دو تابع و منتظر ماندن تا هر دو تمام شوند
    result1, result2 = await asyncio.gather(
        task_one(),
        task_two()
    )

    print("🎯 Both tasks completed!")
    print("Task1:", result1)
    print("Task2:", result2)
    print("➡ Continuing the rest of the program...")

# اجرای برنامه
asyncio.run(main())
