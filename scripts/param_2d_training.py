import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import random
from shapely import Polygon, Point, intersection
from tqdm import tqdm
from pathlib import Path
import math
import torch 
import torch.nn as nn
import os

from copy import deepcopy as dc
from sklearn.mixture import GaussianMixture

from utils import *
from models import SimpleCNN

ROBOTS_NUM = 12
ROBOT_RANGE = 5.0
GRID_STEPS = 64
TARGETS_NUM = 3
COMPONENTS_NUM = 2
PARTICLES_NUM = 500
AREA_W = 20.0
vmax = 1.5
SAFETY_DIST = 2.0
EPISODES = 1


home = Path().home()
path = home / "liquid_networks/datasets/coverage"
print("Path ", path)
files = [x for x in path.glob("**/*") if x.is_file()]

imgs = []
vels = []
FILES_NUM = len(files)//2

for i in range(FILES_NUM):
  imgs.append(np.load(str(path / f"test{i}.npy")))
  vels.append(np.load(str(path / f"vels{i}.npy")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
#c=0

#a=torch.tensor(imgs[1])
#b=torch.tensor(imgs[2])

#c= torch.cat((a,b), dim=0)
#c.shape

# definire c come una matrice vuota
c=torch.empty((0,6,64,64))
for i in range(200):
  a=torch.tensor(imgs[i], dtype=torch.float32)
#a=imgs[i]
  c=torch.cat((c,a), dim=0)

r1= c[:, 0, :, :]
r2= c[:, 1, :, :]
r3= c[:, 2, :, :]
r4= c[:, 3, :, :]
r5= c[:, 4, :, :]
r6= c[:, 5, :, :]

p1 = torch.cat((r1, r2, r3, r4, r5, r6))

# crea tensori anche sulle velocità in uscita
f=torch.empty((0,6,2))

for j in range(200):
  e=torch.tensor(vels[j], dtype=torch.float32)
#a=imgs[i]
  f=torch.cat((f,e), dim=0)

v1= f[:, 0, :]
v2= f[:, 1, :]
v3= f[:, 2, :]
v4= f[:, 3, :]
v5= f[:, 4, :]
v6= f[:, 5, :]

w1 = torch.cat((v1, v2, v3, v4, v5, v6))

x1=p1.unsqueeze(1)

y1=w1.unsqueeze(1)


# Parametrize velocity
resolution = 2*ROBOT_RANGE / GRID_STEPS
y1 /= resolution

# p1 contiene gli ingressi come tensori; w1 contiene le uscite come tensori
# dividere i dati in train e test
import math
#training_data_len = math.ceil(24750 * .8)
# print(training_data_len)

X_train, Y_train= x1, y1
X_train, Y_train = X_train.to(device), Y_train.to(device)

p1.shape, w1.shape, X_train.shape, Y_train.shape #, X_test.shape, Y_test.shape

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, Y_train)
#test_dataset  = TensorDataset(X_test,  Y_test)

train_loader = DataLoader(train_dataset, shuffle=False, batch_size=32)
#test_loader  = DataLoader(test_dataset,  shuffle=False, batch_size=32)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0], batch[1]
    print(x_batch.shape, y_batch.shape)
    break


lr = 0.001
NUM_EPOCHS = 20

cnn_lstm = SimpleCNN().to(device)
from torch import optim
# Loss and optimizer
#criterion = nn.CrossEntropyLoss()
criterion=nn.MSELoss()

optimizer = torch.optim.Adam(cnn_lstm.parameters(), lr=lr)


cnn_lstm.train()

# Train the model
total_step = len(train_loader)
loss_values=[]
for epoch in range(NUM_EPOCHS):
  for i, (images, labels) in enumerate(train_loader):
    # clear gradients for this training step
    optimizer.zero_grad()

    output = cnn_lstm(images)
    #labels=labels.squeeze(1)
    loss = criterion(output.unsqueeze(1), labels)

    # backpropagation, compute gradients
    loss.backward()
    # apply gradients
    optimizer.step()

    running_loss =+ loss.item()

        
  print (f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')
  #if (i+1) % 619 == 0:
    # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
    # print(f"Predicted: {output[epoch+1]}")
    #print(f"Lables: {labels[epoch+1]}")


  loss_values.append(running_loss)


save_path = Path().resolve() /'trained_models/2d_param_cnn.pt'
torch.save(cnn_lstm.state_dict(), str(save_path))
print("Model saved!")