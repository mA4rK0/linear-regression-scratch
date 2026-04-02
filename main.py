data = [
    (1, 3),
    (3, 2),
    (4, 7),
    (5, 9),
    (6, 6)
]

X = [x for x, y in data]
Y = [y for x, y in data]

b0 = 0.0
b1 = 0.0
alpha = 0.01
n = len(X)

prev_mse = float('inf')

for epoch in range(1000):

    sum_error = 0
    sum_error_x = 0
    sum_squared_error = 0

    for x, y in zip(X, Y):
        y_hat = b0 + b1 * x
        error = y_hat - y
        
        sum_error += error
        sum_error_x += error * x
        sum_squared_error += error ** 2

    # Gradient
    grad_b0 = sum_error / n
    grad_b1 = sum_error_x / n

    # MSE
    mse = sum_squared_error / n

    # Update parameter
    b0 = b0 - alpha * grad_b0
    b1 = b1 - alpha * grad_b1

    if abs(prev_mse - mse) < 0.0001:
        print(f"Stop at epoch {epoch}")
        break

    prev_mse = mse

    if epoch % 10 == 0:
        print(f"epoch {epoch} | MSE: {mse:.4f} | b0: {b0:.4f} | b1: {b1:.4f}")

print("\n=== FINAL ===")
print(f"Final MSE: {mse:.4f}")
print(f"Final b0: {b0:.4f}")
print(f"Final b1: {b1:.4f}")

print("\n=== MODEL ===")
print(f"y = {b0:.4f} + {b1:.4f}x")