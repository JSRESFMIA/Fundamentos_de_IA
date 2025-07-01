# Importamos la biblioteca para gráficos
import matplotlib.pyplot as plt

# Importamos una función para controlar la salida en notebooks (como Jupyter)
from IPython import display

# Activamos el modo interactivo de matplotlib, que permite actualizar gráficos sin bloquear la ejecución del código
plt.ion()

#=========================================
# Función para graficar los puntajes
#=========================================
def plot(scores, mean_scores):
    # Limpia la salida anterior en la notebook (si estás usando Jupyter)
    display.clear_output(wait=True)
    
    # Muestra el gráfico actual
    display.display(plt.gcf())

    # Limpia la figura actual antes de volver a dibujar (para que no se acumulen líneas)
    plt.clf()

    # Título y etiquetas del gráfico
    plt.title('Training...')
    plt.xlabel('Number of Games')  # Eje X: número de juegos
    plt.ylabel('Score')            # Eje Y: puntaje

    # Graficamos las dos listas: los puntajes individuales y el promedio acumulado
    plt.plot(scores)        # Línea con los puntajes de cada juego
    plt.plot(mean_scores)   # Línea con los puntajes promedio

    # Configuramos para que el mínimo del eje Y sea cero (para que no se baje demasiado)
    plt.ylim(ymin=0)

    # Mostramos el último valor de score y de mean_score como texto en la gráfica
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))           # Último score
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1])) # Último promedio

    # Mostramos el gráfico sin bloquear el programa
    plt.show(block=False)

    # Hacemos una pequeña pausa para que se actualice correctamente la gráfica
    plt.pause(.1)


