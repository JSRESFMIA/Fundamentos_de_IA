#================================================================
# EJEMPLO RED NEURONAL CONVOLUCIONAL
#================================================================
# Traducido de Pytorch tutorial 2023
#================================================================
# Serrano Ramirez Javier
# Fundamentos de Inteligencia Artificial
# IPN ESFM
# Junio 2025
#================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#================================
# Configuración del GPU
#================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#=========================
# Hiper-parámetros 
#=========================
num_epochs = 10
batch_size = 4
learning_rate = 0.001

#========================================================
# Transformación: convertir imágenes y normalizarlas
#========================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

#========================================================
# Cargar CIFAR-10: 60000 imágenes en 10 clases
#========================================================
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#=============================
# Mostrar algunas imágenes
#=============================
def imshow(img):
    img = img / 2 + 0.5  
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))

#===============================
# Red neuronal convolucional
#===============================
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # n,6,14,14
        x = self.pool(F.relu(self.conv2(x)))  # n,16,5,5
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))               # n,120
        x = F.relu(self.fc2(x))               # n,84
        x = self.fc3(x)                       # n,10
        return x

model = ConvNet().to(device)

#==================================================
# Función de pérdida y optimizador
#==================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#===============================
# Entrenamiento del modelo
#===============================
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Entrenamiento completo')

#===============================
# Guardar modelo entrenado
#===============================
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

#==============================
# EVALUACION DEL MODELO
#==============================
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(10)]
    n_class_samples = [0 for _ in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Precisión del modelo: {acc:.2f} %')

    for i in range(10):
        acc_i = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Precisión de {classes[i]}: {acc_i:.2f} %')

