import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import random
from shapely import Polygon, Point, intersection
from tqdm import tqdm
from pathlib import Path
import math

from copy import deepcopy as dc
from sklearn.mixture import GaussianMixture

from utils import *
from models import *

ROBOTS_NUM = 6
ROBOT_RANGE = 3.0
TARGETS_NUM = 4
COMPONENTS_NUM = 4
PARTICLES_NUM = 500
AREA_W = 20.0
GRID_STEPS = 64
vmax = 1.5
SAFETY_DIST = 2.0
EPISODES = 1

resolution = 2 * ROBOT_RANGE / GRID_STEPS


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path().resolve()
path = path / "trained_models"
MODEL_PATH = path/'2d_param_cnn.pt'
print("Model Path ", MODEL_PATH)


import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model = SimpleCNN().to(device)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print(model)


for episode in range(EPISODES):
  targets = np.zeros((TARGETS_NUM, 1, 2))
  for i in range(TARGETS_NUM):
    targets[i, 0, 0] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand(1,1)
    targets[i, 0, 1] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand(1,1)

  # plt.plot([-0.5*AREA_W, 0.5*AREA_W], [-0.5*AREA_W, -0.5*AREA_W], c='tab:blue', label="Environment")
  # plt.plot([0.5*AREA_W, 0.5*AREA_W], [-0.5*AREA_W, 0.5*AREA_W], c='tab:blue')
  # plt.plot([0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, 0.5*AREA_W], c='tab:blue')
  # plt.plot([-0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, -0.5*AREA_W], c='tab:blue')
  # plt.scatter(targets[:, :, 0], targets[:, :, 1], c='tab:orange', label="Targets")
  # # plt.legend()
  # plt.show()

  STD_DEV = 2.0
  samples = np.zeros((TARGETS_NUM, PARTICLES_NUM, 2))
  for k in range(TARGETS_NUM):
    for i in range(PARTICLES_NUM):
      samples[k, i, :] = targets[k, 0, :] + STD_DEV * np.random.randn(1, 2)

  # Fit GMM
  samples = samples.reshape((TARGETS_NUM*PARTICLES_NUM, 2))
  print(samples.shape)
  gmm = GaussianMixture(n_components=COMPONENTS_NUM, covariance_type='full', max_iter=1000)
  gmm.fit(samples)

  means = gmm.means_
  covariances = gmm.covariances_
  mix = gmm.weights_

  print(f"Means: {means}")
  print(f"Covs: {covariances}")
  print(f"Mix: {mix}")


  ## -------- Generate decentralized probability grid ---------
  GRID_STEPS = 64
  s = AREA_W/GRID_STEPS     # step

  xg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
  yg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
  Xg, Yg = np.meshgrid(xg, yg)
  Xg.shape
  print(Xg.shape)

  Z = gmm_pdf(Xg, Yg, means, covariances, mix)
  Z = Z.reshape(GRID_STEPS, GRID_STEPS)
  Zmax = np.max(Z)
  Z = Z / Zmax

  fig, [ax, ax2] = plt.subplots(1, 2, figsize=(12,6))
  plot_occgrid(Xg, Yg, Z, ax=ax)
  plot_occgrid(Xg, Yg, Z, ax=ax2)



  # ---------- Simulate episode ---------
  # ROBOTS_NUM = np.random.randint(6, ROBOTS_MAX)
  ROBOTS_NUM = 12
  converged = False
  NUM_STEPS = 20
  points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, 2)
  robots_hist = np.zeros((1, points.shape[0], points.shape[1]))
  robots_hist[0, :, :] = points
  vis_regions = []
  discretize_precision = 0.5

  imgs = np.zeros((1, ROBOTS_NUM, GRID_STEPS, GRID_STEPS))
  vels = np.zeros((1, ROBOTS_NUM, 2))

  r_step = 2 * ROBOT_RANGE / GRID_STEPS
  for s in range(1, NUM_STEPS+1):
    print(f"*** Step {s} ***")
    row = 0
    if s > 5:
      row = 1

    # mirror points across each edge of the env
    dummy_points = np.zeros((5*ROBOTS_NUM, 2))
    dummy_points[:ROBOTS_NUM, :] = points
    mirrored_points = mirror(points)
    mir_pts = np.array(mirrored_points)
    dummy_points[ROBOTS_NUM:, :] = mir_pts

    # Voronoi partitioning
    vor = Voronoi(dummy_points)

    conv = True
    lim_regions = []
    img_s = np.zeros((ROBOTS_NUM, GRID_STEPS, GRID_STEPS))
    vel_s = np.zeros((ROBOTS_NUM, 2))
    for idx in range(ROBOTS_NUM):
      # Save grid
      p_i = vor.points[idx]
      xg_i = np.linspace(-ROBOT_RANGE, ROBOT_RANGE, GRID_STEPS)
      yg_i = np.linspace(-ROBOT_RANGE, ROBOT_RANGE, GRID_STEPS)
      Xg_i, Yg_i = np.meshgrid(xg_i, yg_i)
      Z_i = gmm_pdf(Xg_i, Yg_i, means-p_i, covariances, mix)
      Z_i = Z_i.reshape(GRID_STEPS, GRID_STEPS)
      Zmax_i = np.max(Z_i)
      Z_i = Z_i / Zmax_i

      neighs = np.delete(vor.points[:ROBOTS_NUM], idx, 0)
      local_pts = neighs - p_i

      # Remove undetected neighbors
      undetected = []
      for i in range(local_pts.shape[0]):
        if local_pts[i, 0] < -ROBOT_RANGE or local_pts[i, 0] > ROBOT_RANGE or local_pts[i, 1] < -ROBOT_RANGE or local_pts[i, 1] > ROBOT_RANGE:
          undetected.append(i)

      local_pts = np.delete(local_pts, undetected, 0)

      img_i = dc(Z_i)
      for i in range(GRID_STEPS):
        for j in range(GRID_STEPS):
          # jj = GRID_STEPS-1-j
          p_ij = np.array([-ROBOT_RANGE+j*r_step, -ROBOT_RANGE+i*r_step])
          # print(f"Point ({i},{j}): {p_ij}")
          for n in local_pts:
            if np.linalg.norm(n - p_ij) <= SAFETY_DIST:
              img_i[i, j] = -1.0

          # Check if outside boundaries
          p_w = p_ij + p_i
          if p_w[0] < -0.5*AREA_W or p_w[0] > 0.5*AREA_W or p_w[1] < -0.5*AREA_W or p_w[1] > 0.5*AREA_W:
            img_i[i, j] = -1.0

      img_in = torch.from_numpy(img_i).unsqueeze(0).unsqueeze(0)
      img_in = img_in.to(torch.float).to(device)
      vel_i = model(img_in) * resolution
      # print(f"Velocity of robot {idx}: {vel_i}")
      # print("points[idx] shape: ", points[idx, :].shape)
      points[idx, 0] = points[idx, 0] + vel_i[0, 0]
      points[idx, 1] = points[idx, 1] + vel_i[0, 1]
    
    robots_hist = np.concatenate((robots_hist, np.expand_dims(points, 0)))




for i in range(ROBOTS_NUM):
  ax.plot(robots_hist[:, i, 0], robots_hist[:, i, 1])
  ax.scatter(robots_hist[-1, i, 0], robots_hist[-1, i, 1])

for i in range(ROBOTS_NUM):
  ax2.scatter(robots_hist[-1, i, 0], robots_hist[-1, i, 1])

plt.show()