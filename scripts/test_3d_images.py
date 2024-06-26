import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("/home/mattia/coverage_cnn/datasets/3d_coverage")
files = [x for x in path.glob("**/*") if x.is_file()]

print("Number of files: ", len(files))

imgs = []
vels = []
FILES_NUM = len(files) // 2

for i in range(FILES_NUM):
    imgs.append(np.load(str(path / f"test{i}.npy")))
    vels.append(np.load(str(path / f"vels{i}.npy")))

print("Image shape: ", imgs[0].shape)
print("Vel shape: ", vels[0].shape)

# visualize an exmaple
r = 3
t = 10

img = imgs[1]
vel = vels[1]

G = img[t, r, ::4, ::4, ::4]
v = vel[t, r, :]
G[G >= 0] = G[G >= 0] / G.max()
print("G shape: ", G.shape)
print("G max: ", G.max())
print("G min : ", G.min())

xg = np.linspace(-10, 10, 64//4)
yg = np.linspace(-10, 10, 64//4)
zg = np.linspace(-10, 10, 64//4)
xg, yg, zg = np.meshgrid(xg, yg, zg)
print("Shapes: ", xg.shape, yg.shape, zg.shape)

# print("Vel shape: ", vel.shape)


ax = plt.figure().add_subplot(projection='3d')
ax.scatter(xg, yg, zg, c=G, cmap="YlOrRd")
ax.plot([0.0, 2*v[0]], [0.0, 2*v[1]], [0.0, 2*v[2]], c="tab:blue", linewidth=5)
plt.show()