using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Perceptron
{
    // Compuerta Lógica AND
    int[,] datos = {{ 1, 1, 1 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 0 }};

    Random aleatorio = new Random();
    double[] pesos = { aleatorio.NextDouble(), aleatorio.NextDouble(), aleatorio.NextDouble() };
    bool entrenamiento = true;
    int salidaInt;

    while (entrenamiento) 
    {
        entrenamiento = false;

        for (int i = 0; i < datos.GetLength(0); i++) 
        {
            double salidaDoub = datos[i, 0] * pesos[0] + datos[i, 1] * pesos[1] + pesos[2];

            salidaInt = salidaDoub >= 0 
                ? 1 
                : 0;
            
            if (salidaInt != datos[i, 2]) 
            {
                pesos[0] = pesos[0] + 0.1 * (datos[i, 2] - salidaInt) * datos[i, 0];
                pesos[1] = pesos[1] + 0.1 * (datos[i, 2] - salidaInt) * datos[i, 1];
                pesos[2] = pesos[2] + 0.1 * (datos[i, 2] - salidaInt);
                entrenamiento = true;
            }
        }
    }

    // Fin del entrenamiento
    Console.WriteLine("Pesos finales: " + pesos[0] + ", " + pesos[1] + ", " + pesos[2]);
    Console.ReadLine();

    for (int i = 0; i < datos.GetLength(0); i++) 
    {
        double salidaDoub = datos[i, 0] * pesos[0] + datos[i, 1] * pesos[1] + pesos[2];
        salidaInt = salidaDoub >= 0 
            ? 1 
            : 0;
        
        Console.WriteLine("Entrada: " + datos[i, 0] + "AND " + datos[i, 1] + " Salida: " + salidaInt);
    }

    // Compuerta Lógica XOR
    int[,] datosXOR = { { 1, 1, 0 }, { 1, 0, 1 }, { 0, 1, 1 }, { 0, 0, 0 } };

    for (int i = 0; i < datosXOR.GetLength(0); i++)
    {
        double salidaDoubXOR = datosXOR[i, 0] * pesos[0] + datosXOR[i, 1] * pesos[1] + pesos[2];
        salidaInt = salidaDoubXOR >= 0
            ? 1
            : 0;

        Console.WriteLine("Entrada: " + datosXOR[i, 0] + " XOR " + datosXOR[i, 1] + " Salida: " + salidaInt);
    }
}