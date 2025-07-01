#================================================================
# Introduccion al uso de softmax y cross entropy loss en Pytorch
#================================================================
# Serrano Ramirez Javier
# Fundamentos de Inteligencia Artificial
# IPN ESFM
# Junio 2025
#================================================================

#============================================
# Módulos necesarios
#============================================
import torch
import torch.nn as nn
import numpy as np

#================================
# Modelo de Boltzmann
#================================
# En termodinámica es la PROBABILIDAD de encontrar un sistema en algún
# estado dada su energía y temperatura
#======================================================================
#            ->2.0                ->0.65
# linear     ->1.0      ->Softmax ->0.25  -> CrossEntropy(y, y_hat)
#            ->0.1       -> 0.1
#
# puntajes (logits)               probabilidades
#                                 suma = 1.0
#=======================================================================

#=========================================================================
# Softmax aplica el modelo de distribución exponencial para cada elemento
# normalizada con la suma de todas las exponenciales
#=========================================================================
def softmax(x):
    exp = np.exp(x - np.max(x))  # Evita overflow
    return exp / np.sum(exp)

#===============
# vector en R3
#===============
x = np.array([2.0, 1.0, 0.1])

#================================
# softmax de elementos del vector
#================================
outputs = softmax(x)
print('softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)   # tomar softmax de los elementos en el eje 0
print('softmax torch:', outputs)

#===================================================================================
# Cross-entropy loss, o log loss, mide el rendimiento de un modelo de clasificación
# cuya salida es un valor de probabilidad entre 0 y 1
#===================================================================================
# Se incrementa conforme la probabilidad diverge del valor verdadero
#===================================================================================
def cross_entropy(actual, predicted):
    EPS = 1e-15
    predicted = np.clip(predicted, EPS, 1 - EPS)
    loss = -np.sum(actual * np.log(predicted))
    return loss

#====================================
# y debe ser alguna de las opciones
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
#====================================
Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

#===========================================
# CrossEntropyLoss en PyTorch (aplica Softmax)
# nn.LogSoftmax + nn.NLLLoss
# NLLLoss = "negative log likelihood loss"
#===========================================
loss = nn.CrossEntropyLoss()

#========================================================
# objetivo es de tamaño nSamples = 1
# cada elemento tiene etiqueta de clase: 0, 1 o 2
# Y contiene etiquetas de clase, no opciones binarias
#========================================================
Y = torch.tensor([0])

#============================================================
# input es de tamaño nSamples x nClass = 1 x 3
# y_pred debe estar sin normalizar (logits), no con softmax
#============================================================
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'PyTorch Loss1: {l1.item():.4f}')
print(f'PyTorch Loss2: {l2.item():.4f}')

#=================================================
# predicciones (regresa el máximo en la dimensión)
#=================================================
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y.item()}, Y_pred1: {predictions1.item()}, Y_pred2: {predictions2.item()}')

#==================================================================
# Permite calcular el rendimiento para múltiples conjuntos de datos
#==================================================================
Y = torch.tensor([2, 0, 1])

#=====================================================
# matriz input es de tamaño nBatch x nClass = 3 x 3
# Y_pred son logits (sin softmax)
#=====================================================
Y_pred_good = torch.tensor([
    [0.1, 0.2, 3.9],   # predice clase 2
    [1.2, 0.1, 0.3],   # predice clase 0
    [0.3, 2.2, 0.2]])  # predice clase 1

Y_pred_bad = torch.tensor([
    [0.9, 0.2, 0.1],
    [0.1, 0.3, 1.5],
    [1.2, 0.2, 0.5]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Batch Loss1: {l1.item():.4f}')
print(f'Batch Loss2: {l2.item():.4f}')

#=================================
# Predicciones
#=================================
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Clase verdadera: {Y}, Y_pred1: {predictions1}, Y_pred2: {predictions2}')

#==============================================
# Clasificación binaria (red neuronal)
#==============================================
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        y_pred = torch.sigmoid(out)  # sigmoide al final
        return y_pred

#================================================
# Correr problema de clasificación binaria
#================================================
model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()

#================================
# MULTIPLES CLASES
#================================
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out  # sin softmax al final

#========================================
# Correr problema de MULTIPLES CLASES
#========================================
model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()  # (aplica softmax internamente)


