# Definir los pesos y el umbral del perceptrón
weights = [1, 1]
bias = -1.5


# Definir la función de activación
def activation_function(inputs):
    # Calcular la suma ponderada de las entradas y el sesgo
    weighted_sum = sum(w * x for w, x in zip(weights, inputs)) + bias

    # Aplicar la función de activación (en este caso, una función escalón)
    if weighted_sum >= 0:
        return 1
    else:
        return 0


# Definir los datos de ejemplo
data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

# Probar el perceptrón con los datos de ejemplo
for inputs, expected_output in data:
    output = activation_function(inputs)
    print(
        f"Entradas: {inputs}\n Salida esperada: {expected_output}\n Salida del perceptrón: {output}\n\n"
    )
