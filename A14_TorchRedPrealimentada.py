#================================================================
# Red prealimentada feed forward
#================================================================
# Serrano Ramirez Javier
# Fundamentos de Inteligencia Artificial
# IPN ESFM
# Junio 2025
#================================================================
#==================================
# Módulos necesarios
#=================================
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#=================================
# Configuración del GPU
#=================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#===============================
# Hiper-parámetros
#=================================
input_size = 784           # imagen 28x28
hidden_size = 500          # neuronas ocultas
num_classes = 10           # clasificaciones
num_epochs = 2             # iteraciones sobre los datos
batch_size = 100           # tamaño de conjuntos de datos
learning_rate = 0.001      # tasa de aprendizaje

#======================================
# MNIST base de datos
#======================================
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

#================================
# Carga de datos
#================================
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = next(examples)

#===========================
# Mostrar datos en una imagen
#===========================
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()

#===============================================
# Red neuronal completamente conectada con una capa oculta
#===============================================
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # sin activación y sin softmax al final
        # porque la aplica CrossEntropyLoss
        return out

#===================================
# Correr modelo en el GPU
#===================================
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

#===================================
# OPTIMIZACION y cálculo de error
#===================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#=============================
# Entrenar el modelo
#=============================
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #===========================================
        # dimensiones originales: [100, 1, 28, 28]
        # nuevas dimensiones: [100, 784]
        #===========================================
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        #==============
        # evaluación
        #==============
        outputs = model(images)
        loss = criterion(outputs, labels)

        # cálculo del gradiente y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # diagnóstico
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

#================================================
# Evaluar el modelo
# En fase de prueba no se requieren gradientes
#================================================
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max devuelve (valor, índice)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    # precisión
    acc = 100.0 * n_correct / n_samples
    print(f'Precisión de la red en las 10000 imágenes de prueba: {acc:.2f} %')
