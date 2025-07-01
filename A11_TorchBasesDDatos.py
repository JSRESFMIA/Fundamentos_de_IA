#===========================================
# Manejo de Datos en Pytorch
#===========================================
# Serrano Ramirez Javier
# Fundamentos de Inteligencia Artificial
# IPN ESFM
# Junio 2025
#===========================================

#===========================================
# Modulos necesarios
#===========================================
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# ====================================================
# Bigdata debe dividirse en pequeños grupos de datos
# ====================================================

#===============================================
#  Ciclo de entrenamiento
#  for epoch in range(num epochs):
#    # ciclo sobre todos los grupos de datos
#     for i in range(total_batches)
#==================================================

#=====================================================================================
# epoch = una evaluación y retropropagación para todos los datos de entrenamiento
# total_batches = número total de subconjuntos de datos
# batch_size = número de datos de entrenaminto en cada subconjunto
# number of iteraciones = número de iteraciones sobre todos los datos de entrenamiento
#======================================================================================
# e.g : 100 samples, batch_size=20 -> 100/20=5 iteraciones for i epoch
#======================================================================================

#================================================
# Implementación de base de datos típica
# implement __init__ , __getitem__ ,  and __len__
#================================================

#===================
# Hijo de Dataset
#===================
class WineDataset(Dataset):
      #===============================
      # Inicializar, bajar datos, etc.
      # lectura con numpy o pandas
      #================================
      # típicos datos separados por coma
      # delimiter = símbolo delimitador
      # skiprows = líneas de encabezado
      #================================
    def __init__(self):
        # Leer el archivo CSV correctamente
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 1:])
        self.y_data = torch.from_numpy(xy[:, [0]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# ==============================
# Instanciar Base de Datos
# ==============================
dataset = WineDataset()

# Leer características del primer dato
first_data = dataset[0]
features, labels = first_data
print(features, labels)

# =============================
# Cargar con DataLoader
# =============================
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0)  #0 si tienes problemas en Windows

# Ver un dato al azar
dataiter = iter(train_loader)
data = next(dataiter)
features, labels = data
print(features, labels)

# ============================
# Ciclo de aprendizaje vacío
# ============================
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        if (i + 1) % 5 == 0:
            print(f'Epoch: {epoch + 1}/{num_epochs}, Step {i + 1}/{n_iterations} | Inputs {inputs.shape} | Labels {labels.shape}')

# =============================
# Cargar MNIST con torchvision
# =============================
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=3,
                          shuffle=True)
#===============================
# Ver una muestra aleatoria
#===============================
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)


