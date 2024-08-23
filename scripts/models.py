import numpy as np
import random
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F




class SimpleCNN(nn.Module):
      def __init__(self):
        super(SimpleCNN, self).__init__()
        # Definire i livelli convoluzionali
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Definire i livelli fully connected
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # 8x8 è la dimensione dopo il max pooling
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # supponendo 2 classi di output

      def forward(self, x):
        # Primo livello convoluzionale
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Secondo livello convoluzionale
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Terzo livello convoluzionale
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flattening
        x = x.view(-1, 128 * 8 * 8)

        # Livelli fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

class Multichannel_2D_CNN(nn.Module):
      def __init__(self):
        super(Multichannel_2D_CNN, self).__init__()
        # Definire i livelli convoluzionali
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Definire i livelli fully connected
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # 8x8 è la dimensione dopo il max pooling
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # supponendo 2 classi di output

      def forward(self, x):
        # Primo livello convoluzionale
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Secondo livello convoluzionale
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Terzo livello convoluzionale
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flattening
        x = x.view(-1, 128 * 8 * 8)

        # Livelli fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

class CNN_LSTM_3D_new(nn.Module):
  def __init__(self):
    super(CNN_LSTM_3D_new, self).__init__()
    self.cnn = nn.Sequential(
        nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2),
    )
    # self.fc = nn.Linear(hidden_size, num_classes)
    # self.fc1 = nn.Linear(64*16*16*16, 64)
    # self.fc_out = nn.Linear(64, num_classes)
    self.fc1 = nn.Linear(256*2*2*2, 256)
    # self.fc1 = nn.Linear(64*8*8*8, 256*8)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 3)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()

  def forward(self, x):
    #cnn takes input of shape (batch_size, channels, seq_len)
    out = self.cnn(x)
    # lstm takes input of shape (batch_size, seq_len, input_size)
    # print(f"shape after cnn: {out.shape}")
    # out = out.view(out.shape[0], -1, out.shape[1])
    out = out.view(out.shape[0], -1)
    # print("Reshaped output: ", out.shape)
    # out, _ = self.lstm(out)
    # out = self.fc(out[:, -1, :])
    out = self.relu(self.fc1(out))
    out = self.relu(self.fc2(out))
    out = self.fc3(out)
    
    return out

class CNN_LSTM_3D_new2(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(CNN_LSTM_3D_new2, self).__init__()
    self.cnn = nn.Sequential(
        nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2),
    )
    # self.fc = nn.Linear(hidden_size, num_classes)
    # self.fc1 = nn.Linear(64*16*16*16, 64)
    # self.fc_out = nn.Linear(64, num_classes)
    self.fc1 = nn.Linear(256*2*2*2, 256)
    # self.fc1 = nn.Linear(64*8*8*8, 256*8)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, num_classes)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()

  def forward(self, x):
    #cnn takes input of shape (batch_size, channels, seq_len)
    out = self.cnn(x)
    # lstm takes input of shape (batch_size, seq_len, input_size)
    # print(f"shape after cnn: {out.shape}")
    # out = out.view(out.shape[0], -1, out.shape[1])
    out = out.view(out.shape[0], -1)
    # print("Reshaped output: ", out.shape)
    # out, _ = self.lstm(out)
    # out = self.fc(out[:, -1, :])
    out = self.relu(self.fc1(out))
    out = self.relu(self.fc2(out))
    out = self.fc3(out)
    
    return out
