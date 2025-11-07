import seaborn as sns
import matplotlib.pyplot as plt

# داده‌ی نمونه (نمرات دانش‌آموزان)
scores = [45, 60, 65, 70, 72, 74, 76, 78, 80, 82, 85, 88, 90, 92, 95, 97, 99]

# تنظیم تم گرافیکی
sns.set_theme(style="darkgrid")

# رسم هیستوگرام به همراه منحنی KDE
sns.histplot(scores, kde=True, color="skyblue", bins=8)

# اضافه کردن عنوان و برچسب‌ها
plt.title("Distribution of Students' Scores", fontsize=14)
plt.xlabel("Score")
plt.ylabel("Frequency")

# نمایش نمودار
plt.show()
