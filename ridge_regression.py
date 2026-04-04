data = [
    ([1, 2], 5),
    ([2, 1], 6),
    ([3, 3], 10),
    ([4, 2], 11),
    ([5, 3], 14)
]

X = [x for x, y in data]
Y = [y for x, y in data]

n = len(X)
num_features = len(X[0])

b0 = 0.0
weights = [0.0] * num_features

alpha = 0.01
lambda_ = 0.1
prev_mse = float('inf')

for epoch in range(1000):

    sum_error = 0
    sum_error_w = [0.0] * num_features
    sum_squared_error = 0
    sum_w_square = 0

    for x, y in zip(X, Y):

        y_hat = b0
        for i in range(num_features):
            y_hat += weights[i] * x[i]

        error = y_hat - y

        sum_error += error
        sum_squared_error += error ** 2

        for i in range(num_features):
            sum_error_w[i] += error * x[i]

    for w in weights:
        sum_w_square += w ** 2

    mse = (sum_squared_error / n) + (lambda_ * sum_w_square)

    # gradient
    grad_b0 = sum_error / n

    grad_w = []
    for i in range(num_features):
        grad = (sum_error_w[i] / n) + (lambda_ * weights[i])
        grad_w.append(grad)

    b0 -= alpha * grad_b0

    for i in range(num_features):
        weights[i] -= alpha * grad_w[i]

    if abs(prev_mse - mse) < 0.0001:
        print(f"Stop at epoch {epoch}")
        break

    prev_mse = mse

    if epoch % 10 == 0:
        print(f"epoch {epoch} | MSE: {mse:.4f} | b0: {b0:.4f} | weights: {[round(w,4) for w in weights]}")

print("\n=== FINAL ===")
print(f"Final MSE: {mse:.4f}")
print(f"Final b0: {b0:.4f}")
print(f"Final weights: {weights}")

print("\n=== MODEL ===")
model = f"y = {b0:.4f}"
for i, w in enumerate(weights):
    model += f" + {w:.4f}*x{i+1}"
print(model)