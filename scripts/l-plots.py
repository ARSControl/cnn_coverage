import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.mixture import GaussianMixture

from utils import *

path = Path.home() / "cnn_coverage"
robots = np.load(str(path/"results/elle_pos.npy"))
robots_std = np.load(str(path/"results/stdelle_pos.npy"))
print("Robots shape: ", robots.shape)

ROBOTS_NUM = robots.shape[1]
TARGETS_NUM = 2
PARTICLES_NUM = 500
AREA_W = 30.0

targets = np.zeros((TARGETS_NUM, 1, 2))
targets[0, 0, 0] = 0.0
targets[0, 0, 1] = 12.5
targets[1, 0, 0] = 8.0
targets[1, 0, 1] = 7.5

STD_DEV = 3.0
samples = np.zeros((TARGETS_NUM, PARTICLES_NUM, 2))
for k in range(TARGETS_NUM):
  for i in range(PARTICLES_NUM):
    samples[k, i, :] = targets[k, 0, :] + STD_DEV * np.random.randn(1, 2)



# Fit GMM
samples = samples.reshape((TARGETS_NUM*PARTICLES_NUM, 2))
print(samples.shape)
gmm = GaussianMixture(n_components=TARGETS_NUM, covariance_type='full', max_iter=1000)
gmm.fit(samples)

means = gmm.means_
covariances = gmm.covariances_
mix = gmm.weights_

# print(f"Means: {means}")
# print(f"Covs: {covariances}")
# print(f"Mix: {mix}")


## -------- Generate decentralized probability grid ---------
GRID_STEPS = 64
s = AREA_W/GRID_STEPS     # step

xg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
yg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
Xg, Yg = np.meshgrid(xg, yg)
# Xg.shape
# print(Xg.shape)

Z = gmm_pdf(Xg, Yg, means, covariances, mix)
Z = Z.reshape(GRID_STEPS, GRID_STEPS)
Zmax = np.max(Z)
Z = Z / Zmax
print("Z min: ", Z.min())
print("Z max: ", Z.max())


obstacles = np.array([5.0, -5.0])
OBS_W = 20.0
BW = 5

fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.pcolormesh(Xg, Yg, Z, cmap="Reds", vmin=0.01, alpha=0.5)
# ax.plot([obstacles[0]-0.5*OBS_W, obstacles[0]+0.5*OBS_W], [obstacles[1]-0.5*OBS_W, obstacles[1]-0.5*OBS_W], c='tab:blue')
# ax.plot([obstacles[0]+0.5*OBS_W, obstacles[0]+0.5*OBS_W], [obstacles[1]-0.5*OBS_W, obstacles[1]+0.5*OBS_W], c='tab:blue')
# ax.plot([obstacles[0]+0.5*OBS_W, obstacles[0]-0.5*OBS_W], [obstacles[1]+0.5*OBS_W, obstacles[1]+0.5*OBS_W], c='tab:blue')
# ax.plot([obstacles[0]-0.5*OBS_W, obstacles[0]-0.5*OBS_W], [obstacles[1]+0.5*OBS_W, obstacles[1]-0.5*OBS_W], c='tab:blue')
ax.plot([-0.5*AREA_W, obstacles[0]-0.5*OBS_W], [-0.5*AREA_W, -0.5*AREA_W], c='k', lw=BW)
ax.plot([obstacles[0]-0.5*OBS_W, obstacles[0]-0.5*OBS_W], [-0.5*AREA_W, obstacles[1]+0.5*OBS_W], c='k', lw=BW)
ax.plot([obstacles[0]-0.5*OBS_W, 0.5*AREA_W], [obstacles[1]+0.5*OBS_W, obstacles[1]+0.5*OBS_W], c='k', lw=BW)
ax.plot([0.5*AREA_W, 0.5*AREA_W], [obstacles[1]+0.5*OBS_W, 0.5*AREA_W], c='k', lw=BW)
ax.plot([0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, 0.5*AREA_W], c='k', lw=BW)
ax.plot([-0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, -0.5*AREA_W], c='k', lw=BW)


ax.set_xticks([])
ax.set_yticks([])
# ax.plot([0.0, 2*v[0]], [0.0, 2*v[1]], c="tab:blue", linewidth=5)

for i in range(ROBOTS_NUM):
  ax.plot(robots[:, i, 0], robots[:, i, 1], c='tab:blue', lw=3)
  ax.scatter(robots[0, i, 0], robots[0, i, 1], facecolors='none', edgecolors='tab:blue', s=90)
  ax.scatter(robots[-1, i, 0], robots[-1, i, 1], c='tab:blue', s=90)

plt.show()


fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.pcolormesh(Xg, Yg, Z, cmap="Reds", vmin=0.01, alpha=0.5)
# ax.plot([obstacles[0]-0.5*OBS_W, obstacles[0]+0.5*OBS_W], [obstacles[1]-0.5*OBS_W, obstacles[1]-0.5*OBS_W], c='tab:blue')
# ax.plot([obstacles[0]+0.5*OBS_W, obstacles[0]+0.5*OBS_W], [obstacles[1]-0.5*OBS_W, obstacles[1]+0.5*OBS_W], c='tab:blue')
# ax.plot([obstacles[0]+0.5*OBS_W, obstacles[0]-0.5*OBS_W], [obstacles[1]+0.5*OBS_W, obstacles[1]+0.5*OBS_W], c='tab:blue')
# ax.plot([obstacles[0]-0.5*OBS_W, obstacles[0]-0.5*OBS_W], [obstacles[1]+0.5*OBS_W, obstacles[1]-0.5*OBS_W], c='tab:blue')
ax.plot([-0.5*AREA_W, obstacles[0]-0.5*OBS_W], [-0.5*AREA_W, -0.5*AREA_W], c='k', lw=BW)
ax.plot([obstacles[0]-0.5*OBS_W, obstacles[0]-0.5*OBS_W], [-0.5*AREA_W, obstacles[1]+0.5*OBS_W], c='k', lw=BW)
ax.plot([obstacles[0]-0.5*OBS_W, 0.5*AREA_W], [obstacles[1]+0.5*OBS_W, obstacles[1]+0.5*OBS_W], c='k', lw=BW)
ax.plot([0.5*AREA_W, 0.5*AREA_W], [obstacles[1]+0.5*OBS_W, 0.5*AREA_W], c='k', lw=BW)
ax.plot([0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, 0.5*AREA_W], c='k', lw=BW)
ax.plot([-0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, -0.5*AREA_W], c='k', lw=BW)


ax.set_xticks([])
ax.set_yticks([])
# ax.plot([0.0, 2*v[0]], [0.0, 2*v[1]], c="tab:blue", linewidth=5)

for i in range(ROBOTS_NUM):
  ax.plot(robots_std[:, i, 0], robots_std[:, i, 1], c='tab:blue', lw=3)
  ax.scatter(robots_std[0, i, 0], robots_std[0, i, 1], facecolors='none', edgecolors='tab:blue', s=90)
  ax.scatter(robots_std[-1, i, 0], robots_std[-1, i, 1], c='tab:blue', s=90)

th = np.arange(0, 2*np.pi+np.pi/20, np.pi/20)
R = 2.5
xc = obstacles[0]-0.5*OBS_W + R*np.cos(th)
yc = obstacles[1]+0.5*OBS_W + R*np.sin(th)
# ax.plot(xc, yc, lw=7, c='r')

plt.show()


"""
figpath = Path.home() / "cnn_coverage/pics"
plt.savefig(figpath/"l-results.png")
plt.show()
"""