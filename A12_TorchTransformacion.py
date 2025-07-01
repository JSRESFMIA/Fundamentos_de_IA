#===========================================
# TRANSFORMACIONES DE TENSORES
#===========================================
# Serrano Ramirez Javier
# Fundamentos de Inteligencia Artificial
# IPN ESFM
# MAYO 2025
#===========================================
'''
Transformaciones pueden ser aplicadas a imaganes PIL , tensores , ndarrays
o datos comunes durante la creación de la base de datos

lista completa de transformaciones ya programadas:
https://pytorch.org/docs/stable/torchvision/transforms.html

En imágenes
===========
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

En Tensores
===========
LinearTransformation , Normalize,Random,BandomErasing

Conversiones
============
ToPILImage: convertir de tensor a numpy ndrarray
ToTensor : de numpy.ndarray a PILImage

Generico
========
Usar Lambda

Comunes
========
Escribir tu propio objeto (clase)
Componer (compose) múltiples transformaciones
===========================================
composed = transforms.Compose([Rescale(256), RandomCrop(224)])
'''
#=================================
#  Módulos Necesarios
#=================================
import torch 
import torchvision
from torch.utils.data import Dataset
import numpy as np

#=====================================
#  Clase WineDataset hija de Dataset
#=====================================
class WineDataset(Dataset):
    #=================================
    #  CONSTRUCTOR
    #=================================
    def __init__(self, transform=None):
        xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        #=========================================
        # NO SE CONVIERTE EN TENSOR AQUI
        #=========================================
        
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]

        self.transform = transform

    #=================================
    #  Método para obtener un dato
    #=================================
    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    #=================================
    #  Tamaño de conjunto de datos
    #=================================
    def __len__(self):
        return self.n_samples

#=================================
#  Transformaciones comunes
#=================================

#=============================
# De numpy a tensor pytorch
#=============================
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
#===================================================
# Escalar datos (multiplicarlos por una constante)
#===================================================
class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

#=================================
#  PROGRAMA PRINCIPAL
#=================================
if __name__ == "__main__":
    print('Sin transformación')
    dataset = WineDataset()
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features, labels)

    print('\nTransformado en tensor')
    dataset = WineDataset(transform=ToTensor())
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features, labels)

    print('\nCon transformación a tensor y multiplicación')
    composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
    dataset = WineDataset(transform=composed)
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features, labels)
