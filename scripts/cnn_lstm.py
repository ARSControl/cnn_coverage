import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import random


path = Path("/home/mattia/coverage_cnn/datasets/coverage")
files = [x for x in path.glob("**/*") if x.is_file()]
print("Number of files: ", len(files))


## Load data
imgs = []
vels = []
FILES_NUM = len(files)//2

for i in range(FILES_NUM):
    imgs.append(np.load(str(path / f"test{i}.npy")))
    vels.append(np.load(str(path / f"vels{i}.npy")))

print(f"Shape of images: {imgs[0].shape}")
print(f"Shape of vels: {vels[0].shape}")


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

train_size = int(p1.shape[0]*0.75)
print("Training size: ", train_size)

X_train, Y_train, X_test, Y_test = p1[:train_size], w1[:train_size], p1[train_size:], w1[train_size:]
X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)
Y_train = Y_train.unsqueeze(1)
Y_test = Y_test.unsqueeze(1)
print(f"Train/Test shapes: {X_train.shape}, {Y_train.shape}, {X_test.shape}, {Y_test.shape}")

## HYPERPARAMS
input_size = 64
output_size = 2
NUM_EPOCHS = 20
learning_rate = 0.001

sequence_length = 64
hidden_size = 128
num_layers = 3
batch_size = 32

loss_hist = []
val_loss_hist = []




train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class myCNN(nn.Module):
    def __init__(self, in_size, out_size):
        super(myCNN, self).__init__()

        self.input_size = in_size
        # self.hidden_size = hidden_size
        self.output_size = out_size

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)     #(16, 64, 64)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)     #(32, 64, 64)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)     #(64, 64, 64)

        self.pool = nn.MaxPool2d(2, 2)          # Downsample by factor 2

        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, self.output_size)

        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.pool(self.activation(self.conv3(x)))

        # print(f"Shape after cnn: {x.shape}")

        # Flatten the tensor
        x = x.view(-1, 64 * 8 * 8)      # [batch_size, 64*8*8]
        # print(f"Shape after view: {x.shape}")



        # Fully connected layers with ReLU
        x = self.activation(self.fc1(x))

        # Output Layer
        x = self.fc2(x)

        return x


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        #cnn takes input of shape (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
X_train, Y_train, X_test, Y_test = X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)


# model = myCNN(input_size, output_size)
model = CNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

cnn_lstm.train()

# Train the model
total_step = len(train_loader)
loss_values=[]
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
            # clear gradients for this training step
            optimizer.zero_grad()

            output = cnn_lstm(images)
            #labels=labels.squeeze(1)
            loss = criterion(output, labels)

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            running_loss =+ loss.item()

            if (i+1) % 619 == 0:
              print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
            #if (i+1) % 619 == 0:
             # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
             # print(f"Predicted: {output[epoch+1]}")
              #print(f"Lables: {labels[epoch+1]}")


  loss_values.append(running_loss)

    ### Testing
    model.eval()
    for i, (test_images, test_labels) in enumerate(test_loader):
        with torch.inference_mode():
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_out = model(test_images)
            test_loss = criterion(test_out, test_labels.squeeze(1))

    loss_hist.append(loss.item())
    val_loss_hist.append(test_loss.item())
    
    print(f"Epoch: {epoch+1} | Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

plt.plot(loss_hist, label="Loss")
plt.plot(val_loss_hist, label="Val loss")
plt.legend()
plt.show()

# for i in range(10):
#     rnd = np.random.randint(0, X_test.shape[0])
#     img = X_test[rnd]
#     label = Y_test[rnd]
#     vel = model(img)
    # print(f"Epoch {i+1} | Predicted: {vel} | Label: {label}")
