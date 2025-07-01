#============================================
# Red Neuronal profunda propia
#============================================
# ReLU en capas intermedias
# Softmax a la salida 
#============================================
# Serrano Ramirez Javier
# Fundamentos de Inteligencia Artificial 
# ESFM IPN 
# Marzo 2025
#========================================================
# Tuve que hacer unas modificaciones ya que no ejecutaba 
#========================================================
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#=====================================
# Valores iniciales al azar
# Dimensiones de cada capa (mm, nn)
#======================================
def init_params():
    W = []
    b = []
    mm = np.array([784, 10, 15, 12])
    nn = np.array([10, 15, 12, 10])
    capas = len(mm)
    print("Red de perceptrones:")
    for i in range(capas):
        print(mm[i], "x", nn[i])
        W.append(np.random.rand(nn[i], mm[i]) - 0.5)
        b.append(np.random.rand(nn[i], 1) - 0.5)
    return W, b

#==========================
# Función ReLU
#==========================
def ReLU(Z):
    return np.maximum(Z, 0)

#==========================
# Función Softmax
#==========================
def softmax(Z):
    expZ = np.exp(Z - np.max(Z))  # estabilidad numérica
    return expZ / np.sum(expZ, axis=0, keepdims=True)

#=============================================
# Propagación hacia adelante (evaluar la red)
#=============================================
def forward_prop(W, b, X):
    A = []
    Z = []
    AA = X
    A.append(AA)
    for i in range(len(W)):
        ZZ = W[i].dot(AA) + b[i]
        if i < len(W)-1:
            AA = ReLU(ZZ)
        else:
            AA = softmax(ZZ)
        Z.append(ZZ)
        A.append(AA)
    return Z, A

#==========================
# Derivada de ReLU
#==========================
def ReLU_deriv(Z):
    return Z > 0

#=================================================================================
# One-hot encoding transforma una etiqueta real a un vector de comparacion one_hot
#=================================================================================
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

#===================================================
# Backpropagation o calculo numerico del gradiente
#===================================================
def backward_prop(Z, A, W, X, Y):
    dW = []
    db = []
    m = X.shape[1]
    n = len(W) - 1
    one_hot_Y = one_hot(Y)

    dZ = A[-1] - one_hot_Y
    dWW = (1 / m) * dZ.dot(A[n].T)
    dbb = np.expand_dims(np.sum(dZ, axis=1) / m, axis=1)
    dW.append(dWW)
    db.append(dbb)

    for i in range(n - 1, -1, -1):
        dZ = W[i + 1].T.dot(dZ) * ReLU_deriv(Z[i])
        dWW = (1 / m) * dZ.dot(A[i].T)
        dbb = np.expand_dims(np.sum(dZ, axis=1) / m, axis=1)
        dW.append(dWW)
        db.append(dbb)

    dW.reverse()
    db.reverse()
    return dW, db

#==========================
# Actualización de pesos
#==========================
def update_params(W, b, dW, db, alpha):
    for i in range(len(W)):
        W[i] -= alpha * dW[i]
        b[i] -= alpha * db[i]
    return W, b

#==========================
# Predicción
#==========================
def get_predictions(A2):
    return np.argmax(A2, axis=0)

#==========================
# Precisión
#==========================
def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

#==========================
# Descenso de gradiente
#==========================
def gradient_descent(X, Y, alpha, iterations):
    W, b = init_params()
    for epoch in range(iterations):
        Z, A = forward_prop(W, b, X)
        dW, db = backward_prop(Z, A, W, X, Y)
        W, b = update_params(W, b, dW, db, alpha)
        if epoch % 10 == 0:
            predictions = get_predictions(A[-1])
            acc = get_accuracy(predictions, Y)
            print(f"Iteración {epoch} - Precisión: {acc*100:.2f}%")
    return W, b

#==========================
# Función auxiliar de predicción
#==========================
def make_predictions(X, W, b):
    _, A = forward_prop(W, b, X)
    return get_predictions(A[-1])

#==========================
# Mostrar imagen con predicción
#==========================
def test_prediction(index, W, b, X_train, Y_train):
    current_image = X_train[:, index, None]
    prediction = make_predictions(current_image, W, b)
    label = Y_train[index]
    print("Prediction:", prediction)
    print("Label:", label)
    plt.imshow(current_image.reshape(28, 28) * 255, cmap='gray')
    plt.axis('off')
    plt.show()

#==========================
# Programa principal
#==========================
if __name__ == "__main__":
    data = pd.read_csv('train.csv')
    data = np.array(data)
    m, n = data.shape
    print("Número de imágenes:", m)
    print("Número de píxeles:", n)
    np.random.shuffle(data)

    data_dev = data[:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n] / 255.

    data_train = data[1000:].T
    Y_train = data_train[0]
    X_train = data_train[1:n] / 255.

    _, m_train = X_train.shape

    # Entrenar la red
    W, b = gradient_descent(X_train, Y_train, alpha=0.15, iterations=100)

    # Probar con algunas imágenes
    for i in range(10):
        test_prediction(i, W, b, X_train, Y_train)

