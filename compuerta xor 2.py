import numpy as np


class Perceptron:
    def _init_(self, input_size, learning_rate=0.1, epochs=1000):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(input_size + 1)

    def activate(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activate(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


# Datos de entrada y etiquetas para la puerta XOR
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 0])

# Crear y entrenar el perceptrón
perceptron = Perceptron(input_size=2)
perceptron.train(training_inputs, labels)

# Probar el perceptrón entrenado
print("Resultado de la puerta XOR:")
for inputs in training_inputs:
    print(f"Input: {inputs}, Output: {perceptron.predict(inputs)}")
