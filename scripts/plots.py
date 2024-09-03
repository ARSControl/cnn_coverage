import numpy as np
import random
from pathlib import Path
import math
import os, sys
from copy import deepcopy as dc
from sklearn.mixture import GaussianMixture

from datetime import datetime

# import custom lib
from utils import *
from models import *



ROBOTS_NUM = 8
ROBOT_RANGE = 2.0
AREA_W = 10.0
AREA_H = 6.0
TARGETS_NUM = 3
SAMPLES_NUM = 100
STD_DEV = 1.0
MAX_STEPS = 200
SAFETY_DIST = 1.0
CONVERGENCE_TOLERANCE = 0.1
NUM_OBSTACLES = 1
NUM_CHANNELS = 3
GAMMA = 0.5
vmax = 1.5
USE_CBF = True
SAVE_POS = True

AREA_BOTTOM = -3.5
AREA_TOP = 2.5

targets = np.array([[3.5, 1.0], [2.5, 2.0], [0.5, 1.0]])

obstacles = np.array([[-1.5, 1.0]])

print("Targets shape:  ", targets.shape)
print("TArget 0: ", targets[0])
print("Target 1: ", targets[1])
samples = np.zeros((TARGETS_NUM, SAMPLES_NUM, 2))
stds = [1.25, 1.5, 0.75]
for k in range(TARGETS_NUM):
    samples[k, :, :] = np.random.normal(loc=targets[k], scale=stds[k], size=(SAMPLES_NUM, 2))

# Fit GMM
samples = samples.reshape((TARGETS_NUM*SAMPLES_NUM, 2))
gmm = GaussianMixture(n_components=TARGETS_NUM, covariance_type="full", max_iter=1000)
gmm.fit(samples)
means = gmm.means_
covariances = gmm.covariances_
mix = gmm.weights_

## -------- Generate decentralized probability grid ---------
GRID_STEPS = 64
s = AREA_W/GRID_STEPS     # step
r_step = 2 * ROBOT_RANGE / GRID_STEPS


xg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
yg = np.linspace(AREA_BOTTOM, AREA_TOP, GRID_STEPS)
Xg, Yg = np.meshgrid(xg, yg)
Xg.shape
print(Xg.shape)

Z = gmm_pdf(Xg, Yg, means, covariances, mix)
Z = Z.reshape(GRID_STEPS, GRID_STEPS)
Zmax = np.max(Z)
Z = Z / Zmax

path = Path.home() / "crazyswarm/ros_ws/src/cnn_coverage"
robots = np.load(path / "results/vid3.npy")
print("robot shape: ", robots.shape)

fig, ax = plt.subplots(1, 1)
plot_occgrid(Xg, Yg, Z, ax=ax)
th = np.arange(0, 2*np.pi+np.pi/20, np.pi/20)
for i in range(ROBOTS_NUM):
    ax.plot(robots[:, i, 0], robots[:, i, 1])
    ax.scatter(robots[-1, i, 0], robots[-1, i, 1], s=18, marker='^')
for obs in obstacles:
    xc = obs[0] + SAFETY_DIST*np.cos(th)
    yc = obs[1] + SAFETY_DIST*np.sin(th)
    ax.plot(xc, yc, linewidth=3, c='b')

ax.set_xlim([-0.5*AREA_W, 0.5*AREA_W])
ax.set_ylim([AREA_BOTTOM, AREA_TOP])

plt.show()

