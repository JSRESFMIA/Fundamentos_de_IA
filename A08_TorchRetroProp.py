#=========================================
# Gradientes simples en Pytorch
#=========================================
# Serrano Ramirez Javier
# Fundamentos de Inteligencia Artificial
# IPN ESFM
# Mayo 2025
#=========================================

import torch

x= torch.tensor(1.0)
y= torch.tensor(2.0)

#===================================
# Variable de diferenciasión (d/dw)
#===================================
w= torch.tensor(1.0, requires_grad=True)

#==================================
# evaluación cálculo de costo
#==================================
y_predicted = w * x
loss = (y_predicted - y)**2
print(loss)

#=========================================
# RETROPROPAGACION para calcular gradiente
# w.grad es el gradiente
#========================================
loss.backward()
print(w.grad)

#===========================================
# Nuevos coeficientes (desceso de gradiente)
# Repetir evaluación y RETROPROPAGACION
#=============================================
with torch.no_grad():
    w -= 0.01 * w.grad
w.grad.zero_()
print(w)
