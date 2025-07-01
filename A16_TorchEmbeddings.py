#================================================================
# Embedings con Pytorch y lighthing
#================================================================
# Serrano Ramirez Javier
# Fundamentos de Inteligencia Artificial
# IPN ESFM
# Junio 2025
#================================================================

import torch 
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.uniform import Uniform
from torch.utils.data import TensorDataset, DataLoader
import lightning as L
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

#=============================================
# Crear los datos de entrenamiento de la red
#=============================================
inputs = torch.tensor([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])

labels = torch.tensor([[0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.],
                       [0., 1., 0., 0.]])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

#=================================
# Embedding con Linear
#=================================
class WordEmbeddingWithLinear(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.input_to_hidden = nn.Linear(in_features=4, out_features=2, bias=False)
        self.hidden_to_output = nn.Linear(in_features=2, out_features=4, bias=False)

    def forward(self, input):
        hidden = self.input_to_hidden(input)
        output_values = self.hidden_to_output(hidden)
        return output_values

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = F.mse_loss(output_i, label_i)
        return loss

#===================================
# CREACION DE LA RED
#===================================
modelLinear = WordEmbeddingWithLinear()

#==========================================
# Mostrar parámetros antes del aprendizaje
#==========================================
weights = modelLinear.input_to_hidden.weight.detach().numpy().T
data = {
    "w1": weights[:, 0],
    "w2": weights[:, 1],
    "token": ["Dunas2", "es", "grandiosa", "Godzilla"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)

sns.scatterplot(data=df, x="w1", y="w2")

for i in range(4):
    plt.text(df.w1[i], df.w2[i], df.token[i],
             horizontalalignment='left',
             size='medium',
             color='black',
             weight='semibold')
plt.show()

#=================================
# Entrenamiento
#=================================
trainer = L.Trainer(max_epochs=500)
trainer.fit(modelLinear, train_dataloaders=dataloader)

#====================================
# Graficar después del entrenamiento
#====================================
weights = modelLinear.input_to_hidden.weight.detach().numpy().T
data = {
    "w1": weights[:, 0],
    "w2": weights[:, 1],
    "token": ["Dunas2", "es", "grandiosa", "Godzilla"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)

sns.scatterplot(data=df, x="w1", y="w2")

for i in range(4):
    plt.text(df.w1[i], df.w2[i], df.token[i],
             horizontalalignment='left',
             size='medium',
             color='black',
             weight='semibold')
plt.show()

#==============================================
# Crear embedding final desde pesos entrenados
#==============================================
word_embeddings = nn.Embedding.from_pretrained(modelLinear.input_to_hidden.weight.T)
vocab = {'Dunas2': 0, 'es': 1, 'grandiosa': 2, 'Godzilla': 3}

print(word_embeddings(torch.tensor(vocab['Dunas2'])))

