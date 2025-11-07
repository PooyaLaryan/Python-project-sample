import numpy as np

# تابع فعال‌سازی و مشتق آن
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# مقداردهی اولیه وزن‌ها و biasها
np.random.seed(42)

# معماری شبکه
input_size = 3
hidden1_size = 20
hidden2_size = 30
output_size = 2
learning_rate = 0.01

# وزن‌ها و بایاس‌ها
W1 = np.random.randn(input_size, hidden1_size) * 0.1
b1 = np.zeros((1, hidden1_size))

W2 = np.random.randn(hidden1_size, hidden2_size) * 0.1
b2 = np.zeros((1, hidden2_size))

W3 = np.random.randn(hidden2_size, output_size) * 0.1
b3 = np.zeros((1, output_size))

# داده‌های ورودی و خروجی نمونه
X = np.random.rand(5, input_size)   # ۵ نمونه با ۳ ویژگی
y_true = np.random.rand(5, output_size)

# آموزش شبکه
for epoch in range(1000):
    # ---- Forward ----
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    y_pred = sigmoid(z3)

    # ---- Loss ----
    loss = np.mean((y_true - y_pred) ** 2)

    # ---- Backward ----
    d_loss = 2 * (y_pred - y_true) / y_true.shape[0]

    d_z3 = d_loss * sigmoid_derivative(y_pred)
    d_W3 = np.dot(a2.T, d_z3)
    d_b3 = np.sum(d_z3, axis=0, keepdims=True)

    d_a2 = np.dot(d_z3, W3.T)
    d_z2 = d_a2 * sigmoid_derivative(a2)
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(a1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # ---- به‌روزرسانی وزن‌ها ----
    W3 -= learning_rate * d_W3
    b3 -= learning_rate * d_b3
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

# تست خروجی نهایی
print("\nخروجی نهایی شبکه:")
print(y_pred)

print("---")
