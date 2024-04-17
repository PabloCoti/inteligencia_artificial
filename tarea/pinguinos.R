# Se requiere la instalacion de los siguientes paquetes
# Install.packages(c("tidyverse", "caret", "neuralnet"))
# carga de paquetes
library(tidyverse)
library(caret)
library(neuralnet)
library(palmerpenguins)
library(ggplot2)
# Carga de datos
datos = penguins
datos <- na.omit(datos)
# Separacion en grupo de entrenamiento y pruebas
muestra = createDataPartition(datos$species, p=0.7, list = F)
train = datos[muestra,]
test = datos[-muestra,]
# Entrenamiento de la red neuronal
red.neuronal = neuralnet(species ~ bill_length_mm + bill_depth_mm +
                           flipper_length_mm + body_mass_g, data = train, hidden = c(2,3))
red.neuronal$act.fct
# Graficacion de la red neuronal
plot(red.neuronal)
# Aplicar el conjunto de pruebas a la red para predecir la especie
prediccion = predict(red.neuronal, test, type = "class")
# Decodificar la columna que contiene el valor maximo
# y por ende la especie de la que se trata
decodificarCol = apply(prediccion, 1, which.max)
# crear una columna nueva con los valores decodificados
codificado <- data.frame(decodificarCol)
codificado <- mutate(codificado, especie = recode(decodificarCol, 
                                                  "1" = "Adelie", 
                                                  "2" = "Chinstrap", 
                                                  "3" = "Gentoo"))

test$Especie.Pred = codificado$especie


datos %>%
  count(species)

datos %>%
  group_by(species) %>%
  summarize(across(where(is.numeric),mean, na.rm=TRUE))

plot(datos)

datos%>%
  count(species) %>%
  ggplot() + geom_col(aes(x = species, y = n, fill = species)) + geom_label(aes(x = species, y = n, label = n )) +
  scale_fill_manual(values = c("darkorange", "purple", "cyan4")) + theme_minimal() + labs(title = "distribucion de pinguinos por especie")
  
  
datos %>%
  drop_na() %>%
  count(sex, species) %>%
  ggplot() + geom_col(aes(x=species, y=n, fill=species)) + geom_label(aes(x=species, y=n, label=n)) +
  scale_fill_manual(values = c("darkorange", "purple", "cyan4")) + facet_wrap(~sex) + theme_minimal() + labs(tittle='Especie de pinguinos ~sexo')

datos %>%
  select_if(is.numeric) %>%
  drop_na %>%
  cor()
  

##clase 22/04/2024
ggplot(data = penguins,
       aes(x=flipper_length_mm,
           y = body_mass_g)) + 
  geom_point(aes(color = species, 
                 shape = species),
             size = 3,
             alpha = 0.8) +
  #theme_minimal ()+
  scale_color_manual(values = c("darkorange", "purple", "cyan4")) +
  labs(title = "Tamaño del pinguino, Palmer Station LTER",
       subtitle = "Longitud de aleta y Masa corporpal para Pinguinos Adelie, Chinstrap, gentoo",
       x = "Longitud de aleta (mm)",
       y = "Masa Corporal (g)",
       color = "Especie de Pinguino",
       shape = "Especie de Pinguino") + 
  theme_minimal()

##Grafico 2
ggplot(data = penguins,
       aes(x=flipper_length_mm,
           y = body_mass_g)) + 
  geom_point(aes(color = island, 
                 shape = species),
             size = 3,
             alpha = 0.8) +
  #theme_minimal ()+
  scale_color_manual(values = c("darkorange", "purple", "cyan4")) +
  labs(title = "Tamaño del pinguino, Palmer Station LTER",
       subtitle = "Longitud de aleta y Masa corporpal segun cada isla",
       x = "Longitud de aleta (mm)",
       y = "Masa Corporal (g)",
       color = "Especie de Pinguino",
       shape = "Especie de Pinguino") + 
  theme_minimal()

ggplot(data = penguins,
       aes(x=flipper_length_mm,
           y = bill_length_mm)) + 
  geom_point(aes(color = species, 
                 shape = species),
             size = 3,
             alpha = 0.8) +
  #theme_minimal ()+
  scale_color_manual(values = c("darkorange", "purple", "cyan4")) +
  labs(title = "Tamaño del pinguino, Palmer Station LTER",
       subtitle = "Longitud de aleta y pico segun cada especie",
       x = "Longitud de aleta (mm)",
       y = "Longitud de pico",
       color = "Especie de Pinguino",
       shape = "Especie de Pinguino") + 
  theme_minimal()

datos = penguins
datos <- na.omit(penguins)

# Escalar los datos
datos_scaled <- datos
datos_scaled$bill_length_mm <- scale(datos_scaled$bill_length_mm)
datos_scaled$flipper_length_mm <- scale(datos_scaled$flipper_length_mm)
datos_scaled$bill_depth_mm <- scale(datos_scaled$bill_depth_mm)
datos_scaled$body_mass_g <- scale(datos_scaled$body_mass_g)

# Separación en grupo de entrenamiento y pruebas
muestra <- createDataPartition(datos$species, p = 0.7, list = FALSE)
train <- datos_scaled[muestra, ]
test <- datos_scaled[-muestra, ]

# Entrenamiento de la red neuronal
red.neuronal <- neuralnet(species ~ bill_length_mm + flipper_length_mm + bill_depth_mm + body_mass_g, 
                          data = train,
                          hidden = c(2,3))
red.neuronal$act.fct
# Graficacion de la red neuronal
plot(red.neuronal)

# Aplicar el conjunto de pruebas a la red para predecir la especie
prediccion <- predict(red.neuronal, test, type = "class")

# Decodificar las predicciones
decodificarCol <- apply(prediccion, 1, which.max)
codificado <- data.frame(decodificarCol)
codificado <- mutate(codificado, especie = recode(decodificarCol, 
                                                  "1" = "Adelie", 
                                                  "2" = "Chinstrap", 
                                                  "3" = "Gentoo"))

test$Especie.Pred <- codificado$especie