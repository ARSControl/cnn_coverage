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