import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path.home() / "cnn_coverage/results"
loss = np.load(path/"loss.npy")
print("Loss shape: ", loss.shape)

plt.rcParams.update({'font.size': 24})


fig, ax = plt.subplots(1, 1)
ax.plot(loss[:, 0], lw=5, label="Train Loss")
ax.plot(loss[:, 1], lw=5, label="Test Loss")
ax.set_xlabel("Epochs")
ax.grid()
ax.legend()
savepath = Path.home() / "pics"
plt.savefig(str(savepath/"loss.pdf"))
plt.show()