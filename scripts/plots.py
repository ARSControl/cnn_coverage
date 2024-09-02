import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("/home/mattia/cnn_coverage/dataset_3ch")
files = [x for x in path.glob("**/*") if x.is_file()]

print("Number of files: ", len(files))

imgs = []
vels = []
FILES_NUM = len(files) // 2

for i in range(2):
    imgs.append(np.load(str(path / f"test{i}.npy")))
    vels.append(np.load(str(path / f"vels{i}.npy")))

print("Image shape: ", imgs[0].shape)
print("Vel shape: ", vels[0].shape)




# visualize an exmaple
r = 1
t = 3

img = imgs[1]
vel = vels[1]
GRID_STEPS = img.shape[3]
print("grid steps: ", GRID_STEPS)

G = img[t, r, 0, :, :]
N = img[t, r, 1, :, :]
OBS = img[t, r, 2, :, :]
v = vel[t, r, :]
G = G - np.min(G)
G = G / np.max(G)
print("G shape: ", G.shape)
print("G max: ", G.max())
print("G min : ", G.min())

xg = np.linspace(-5, 5, GRID_STEPS)
yg = np.linspace(-5, 5, GRID_STEPS)
zg = np.linspace(-5, 5, GRID_STEPS)
xg, yg= np.meshgrid(xg, yg)
print("Shapes: ", xg.shape, yg.shape)

# print("Vel shape: ", vel.shape)


fig, axs = plt.subplots(1,3, figsize=(18,6))
axs[0].scatter(xg, yg, c=G, cmap="Greens")
axs[1].scatter(xg, yg, c=N, cmap="Blues")
axs[2].scatter(xg, yg, c=OBS, cmap="Reds")

for ax in axs:
  ax.set_xticks([])
  ax.set_yticks([])
# ax.plot([0.0, 2*v[0]], [0.0, 2*v[1]], c="tab:blue", linewidth=5)ù

figpath = Path.home() / "cnn_coverage/pics"
plt.savefig(figpath/"channels.png")
plt.show()
