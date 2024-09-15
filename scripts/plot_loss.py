import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
from pathlib import Path

path = Path.home() / "cnn_coverage/results"
loss = np.load(path/"loss.npy")
print("Loss shape: ", loss.shape)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 36  })


fig, ax = plt.subplots(1, 1)
ax.plot(loss[:, 0], lw=8, label="Train Loss")
ax.plot(loss[:, 1], lw=8, label="Test Loss")
ax.set_xlabel("Epochs")
ax.grid()
ax.legend()
savepath = Path().resolve() / "pics"
plt.savefig(str(savepath/"loss.pdf"))
plt.show()