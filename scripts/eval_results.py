import numpy as np
import math
from pathlib import Path
from matplotlib import pyplot as plt

path = Path.home() / "cnn_coverage/results"
files = [a for a in path.glob("**/*")]
print("Number of files: ", len(files))

ns = [12, 16, 20]
etas = []
collisions = []
for n in ns:
    etas.append(np.load(path/f"eta{n}.npy"))
    collisions.append(np.load(path/f"collisions{n}.npy"))


for e in range(len(etas)):
    for i in range(etas[e].shape[0]):
        if etas[e][i, 0] == 0:
            break
    etas[e] = etas[e][:i, :]
print("Eta shape after removing: ", etas[0].shape)

for e in range(len(etas)):
    ids = (etas[e]==0).argmax(axis=1)
    print("Id shape: ", ids.shape)
    for i in range(etas[e].shape[0]):
        etas[e][i, ids[i]:] = etas[e][i, ids[i]-1]



etas_m = []
stds = []
for eta in etas:
    etas_m.append(np.mean(eta, axis=0))
    stds.append(np.std(eta, axis=0))

t = np.arange(100)
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, eta in enumerate(etas_m):
    plt.plot(t, eta, label=f"{ns[i]} Robots", c=colors[i])
    plt.fill_between(t, eta-stds[i], eta+stds[i], color=colors[i], alpha=0.2)

plt.grid()
plt.legend()
plt.show()
