# se requiere la instalación de los siguientes paquetes
# install.packages(c("tidyverse", "caret", "nerualnet"))

library(tidyverse)
library(caret)
library(neuralnet)

# Cargar datos de iris en variable datos
datos = iris

# Crear subonjunto para entrenamiento y subconjunto para pruebas
muestra = createDataPartition(datos$Species, p=0.8, list=F)

# al poner solo coma se indica que son todas las columnas
train = datos[muestra,]

# Escoger todos los datos que no estén en la muestra (-muestra)
test = datos[-muestra,]

# Entrenar la red neuronal
red.neuronal = neuralnet(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = train, hidden = c(2, 3))

# Verificar si se entrenó bien por medio de la función de activación
red.neuronal$act.fct

# Graficar red neuronal
plot(red.neuronal)

# Aplicar el conjunto de pruebas a la red neuronal para predecir la especie
prediccion = predict(red.neuronal, test, type="class")

# Decodificar la columna que contiene el valor maximo y por ende la especie de la que se trata decodificarCol
decodificarCol = apply(prediccion, 1, which.max)

# Crear una columna nueva con los valores decodificados
codificado = data_frame(decodificarCol)
codificado = mutate(codificado, especie=recode(
  codificado$decodificarCol, "1"="Setosa", "2"="Versicolor", "3"="Virginica"
))

# Ver el resultado del codificado
view(codificado)

# Agregar columna Especie.Pred a la variable test que es igual a la columna especie de la variable codificado
test$Especie.Pred = codificado$especie
view(test)


