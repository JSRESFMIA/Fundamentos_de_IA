#================================================================
# Algoritmo genetico simple
#================================================================
# Serrano Ramirez Javier
# Fundamentos de Inteligencia Artificial
# IPN ESFM
# Junio 2025
#================================================================
#=================================
#  Módulos necesarios
#=================================
import datetime
import random
#=====================================
# Inicializa la semilla aleatoria
#=====================================
random.seed(random.random())
startTime = datetime.datetime.now()

#=================================
#  Conjunto de genes disponibles
#=================================
geneSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "

#=================================
#  Frase objetivo
#=================================
target = "Hola mundo"

#=================================
#  Generar una cadena inicial aleatoria
#=================================
def generate_parent(length):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    return "".join(genes)

#=================================
#  Función de aptitud
#  Cuenta cuántas letras coinciden con la frase objetivo
#=================================
def get_fitness(guess):
    return sum(1 for expected, actual in zip(target, guess) if expected == actual)

#=================================
#  Mutación: cambia un carácter aleatorio
#=================================
def mutate(parent):
    index = random.randrange(0, len(parent))
    childGenes = list(parent)
    newGene = random.choice(geneSet)
    while newGene == childGenes[index]:
        newGene = random.choice(geneSet)
    childGenes[index] = newGene
    return "".join(childGenes)

#=================================
#  Mostrar intento actual
#=================================
def display(guess):
    timeDiff = datetime.datetime.now() - startTime
    fitness = get_fitness(guess)
    print("{}\t{}\t{}".format(guess, fitness, timeDiff))

#=================================
#  CODIGO PRINCIPAL
#=================================
bestParent = generate_parent(len(target))
bestFitness = get_fitness(bestParent)
display(bestParent)

#=================================
#  Bucle de mejora por MUTACION
#=================================
while True:
    child = mutate(bestParent)
    childFitness = get_fitness(child)
    
    # Si el hijo no mejora, continuar
    if bestFitness >= childFitness:
        display(child)
        continue
    
    display(child)
    
    # Si ya es la frase objetivo, terminar
    if childFitness == len(target):
        break
    
    # Guardar el mejor hasta ahora
    bestFitness = childFitness
    bestParent = child

