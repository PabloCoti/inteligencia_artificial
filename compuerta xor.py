import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def forward(x, w1, w2, predict=False):
    a1 = np.dot(x, w1)
    z1 = sigmoid(a1)

    bias = np.ones((len(z1), 1))
    z1 = np.concatenate((bias, z1), axis=1)
    a2 = np.dot(z1, w2)
    z2 = sigmoid(a2)

    if predict:
        return z2
    return a1, z1, a2, z2


def backprop(a2, z0, z1, z2, y, w2):
    delta2 = z2 - y
    Delta2 = np.dot(z1.T, delta2)
    delta1 = (delta2.dot(w2[1:, :].T)) * sigmoid_deriv(a1)
    Delta1 = np.dot(z0.T, delta1)
    return delta2, Delta1, Delta2


X = np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1]])

y = np.array([[0], [1], [1], [0]])  # etiquetas ajustadas para XOR

np.random.seed(1)
w1 = np.random.randn(3, 5)
w2 = np.random.randn(6, 1)

lr = 0.09
costs = []

epochs = 15000
m = len(X)

for i in range(epochs):
    a1, z1, a2, z2 = forward(X, w1, w2)

    delta2, Delta1, Delta2 = backprop(a2, X, z1, z2, y, w2)

    w1 -= lr * (1 / m) * Delta1
    w2 -= lr * (1 / m) * Delta2

    c = np.mean(np.abs(delta2))
    costs.append(c)


z3 = forward(X, w1, w2, True)
predictions = np.round(z3)
for i in range(len(X)):
    print(f"Entrada: {X[i, 1]} XOR {X[i, 2]} Salida: {int(predictions[i, 0])}")
