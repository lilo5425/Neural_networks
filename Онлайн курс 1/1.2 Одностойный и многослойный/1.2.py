import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([0, 0, 0, 1])

w = np.zeros(2)
b = 0

lr = 0.5
epochs = 10


def step(x):
    return 1 if x >= 0 else 0


for epoch in range(epochs):
    for i in range(len(X)):
        xi = X[i]
        yi = y[i]

        z = np.dot(w, xi) + b
        pred = step(z)

        error = yi - pred
        w += lr * error * xi
        b += lr * error

print("Финальные веса:", w)
print("Финальное смещение:", b)
print("\nПредсказания:")
for i in range(len(X)):
    z = np.dot(w, X[i]) + b
    pred = step(z)
    print(f"  {X[i][0]} AND {X[i][1]} = {pred} (целевое: {y[i]})")
