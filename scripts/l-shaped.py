import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import random
from shapely import Polygon, Point, intersection
from tqdm import tqdm
from pathlib import Path
import math
from scipy.optimize import minimize
import argparse

from copy import deepcopy as dc
from sklearn.mixture import GaussianMixture

from utils import *
from models import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--robots-num', type=int,  default=12, help="number of robots")
  parser.add_argument('--range', type=float,  default=3.0, help="sensing range")
  parser.add_argument('--width', type=float,  default=30.0, help="area width")
  args = parser.parse_args()
  return args

args = parse_args()

ROBOTS_NUM = args.robots_num
ROBOT_RANGE = args.range
TARGETS_NUM = 2
COMPONENTS_NUM = 4
PARTICLES_NUM = 500
AREA_W = args.width
GRID_STEPS = 64
vmax = 1.5
SAFETY_DIST = 2.0
EPISODES = 1
NUM_OBSTACLES = 1
NUM_STEPS = 500
NUM_CHANNELS = 3
GAMMA = 0.5
USE_CBF = False
SAVE_POS = False

resolution = 2 * ROBOT_RANGE / GRID_STEPS


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path().resolve()
path = path / "trained_models"
# MODEL_PATH = path/'multichannel_2d_cnn.pt'
MODEL_PATH = path/'2d_cnn_3ch_cbf.pt'
print("Model Path ", MODEL_PATH)


import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model = Multichannel_2D_CNN(NUM_CHANNELS).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(model)

eval_data = np.zeros((EPISODES, NUM_STEPS))
collision_counter = np.zeros((EPISODES, NUM_STEPS))

def objective_function(u):
  return np.linalg.norm(u)**2

def safety_constraint(u, A, b):
  return -np.dot(A,u) + b

for episode in range(EPISODES):
  print(f"*** Episode {episode} ***")
  targets = np.zeros((TARGETS_NUM, 1, 2))
  targets[0, 0, 0] = 0.0
  targets[0, 0, 1] = 12.5
  targets[1, 0, 0] = 8.0
  targets[1, 0, 1] = 7.5
  # for i in range(TARGETS_NUM):
  #   targets[i, 0, 0] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand()
  #   targets[i, 0, 1] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand()

  # plt.plot([-0.5*AREA_W, 0.5*AREA_W], [-0.5*AREA_W, -0.5*AREA_W], c='tab:blue', label="Environment")
  # plt.plot([0.5*AREA_W, 0.5*AREA_W], [-0.5*AREA_W, 0.5*AREA_W], c='tab:blue')
  # plt.plot([0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, 0.5*AREA_W], c='tab:blue')
  # plt.plot([-0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, -0.5*AREA_W], c='tab:blue')
  # plt.scatter(targets[:, :, 0], targets[:, :, 1], c='tab:orange', label="Targets")
  # # plt.legend()
  # plt.show()

  STD_DEV = 3.0
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

  obstacles = np.array([5.0, -5.0])
  OBS_W = 20.0

  fig, ax = plt.subplots(1, 1, figsize=(8,8))
  plot_occgrid(Xg, Yg, Z, ax=ax)
  ax.plot([obstacles[0]-0.5*OBS_W, obstacles[0]+0.5*OBS_W], [obstacles[1]-0.5*OBS_W, obstacles[1]-0.5*OBS_W], c='tab:blue')
  ax.plot([obstacles[0]+0.5*OBS_W, obstacles[0]+0.5*OBS_W], [obstacles[1]-0.5*OBS_W, obstacles[1]+0.5*OBS_W], c='tab:blue')
  ax.plot([obstacles[0]+0.5*OBS_W, obstacles[0]-0.5*OBS_W], [obstacles[1]+0.5*OBS_W, obstacles[1]+0.5*OBS_W], c='tab:blue')
  ax.plot([obstacles[0]-0.5*OBS_W, obstacles[0]-0.5*OBS_W], [obstacles[1]+0.5*OBS_W, obstacles[1]-0.5*OBS_W], c='tab:blue')
  # ---------- Simulate episode ---------
  # ROBOTS_NUM = np.random.randint(6, ROBOTS_MAX)
  converged = False
  points = np.zeros((ROBOTS_NUM, 2))
  points[:, 0] = -0.5*AREA_W + 8 * np.random.rand(ROBOTS_NUM)
  points[:, 1] = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM)
  robots_hist = np.zeros((1, points.shape[0], points.shape[1]))
  robots_hist[0, :, :] = points
  vis_regions = []
  discretize_precision = 0.5
  dt = 0.25
  failed = False




  imgs = np.zeros((1, ROBOTS_NUM, NUM_CHANNELS, GRID_STEPS, GRID_STEPS))
  vels = np.zeros((1, ROBOTS_NUM, 2))

  r_step = 2 * ROBOT_RANGE / GRID_STEPS
  denom = np.sum(s**2 * gmm_pdf(Xg, Yg, means, covariances, mix))
  print("Total info: ", denom)
  for s in range(1, NUM_STEPS+1):
    # print(f"*** Step {s} ***")
    all_stopped = True

    # mirror points across each edge of the env
    dummy_points = np.zeros((5*ROBOTS_NUM, 2))
    dummy_points[:ROBOTS_NUM, :] = points
    mirrored_points = mirror(points, AREA_W)
    mir_pts = np.array(mirrored_points)
    dummy_points[ROBOTS_NUM:, :] = mir_pts

    # Voronoi partitioning
    vor = Voronoi(dummy_points)

    conv = True
    lim_regions = []
    img_s = np.zeros((ROBOTS_NUM, NUM_CHANNELS, GRID_STEPS, GRID_STEPS))
    vel_s = np.zeros((ROBOTS_NUM, 2))
    num = 0.0
    collision = False
    for idx in range(ROBOTS_NUM):
      p_i = vor.points[idx]


      # Save grid
      xg_i = np.linspace(-ROBOT_RANGE, ROBOT_RANGE, GRID_STEPS)
      yg_i = np.linspace(-ROBOT_RANGE, ROBOT_RANGE, GRID_STEPS)
      Xg_i, Yg_i = np.meshgrid(xg_i, yg_i)
      Z_i = gmm_pdf(Xg_i, Yg_i, means-p_i, covariances, mix)
      Z_i = Z_i.reshape(GRID_STEPS, GRID_STEPS)
      Zmax_i = np.max(Z_i)
      Z_i = Z_i / Zmax_i

      img_i = Z_i * 255
      img_i = np.expand_dims(img_i, 0)

      neighs = np.delete(vor.points[:ROBOTS_NUM], idx, 0)
      local_pts = neighs - p_i

      # Remove undetected neighbors
      undetected = []
      for i in range(local_pts.shape[0]):
        if local_pts[i, 0] < -ROBOT_RANGE or local_pts[i, 0] > ROBOT_RANGE or local_pts[i, 1] < -ROBOT_RANGE or local_pts[i, 1] > ROBOT_RANGE:
          undetected.append(i)

      local_pts = np.delete(local_pts, undetected, 0)

      img_obs = np.zeros((1, GRID_STEPS, GRID_STEPS))
      img_neighs = np.zeros((1, GRID_STEPS, GRID_STEPS))
      for i in range(GRID_STEPS):
        for j in range(GRID_STEPS):
          # jj = GRID_STEPS-1-j
          p_ij = np.array([-ROBOT_RANGE+j*r_step, -ROBOT_RANGE+i*r_step])
          # print(f"Point ({i},{j}): {p_ij}")
          for n in local_pts:
            if np.linalg.norm(n - p_ij) <= SAFETY_DIST:
              img_neighs[0, i, j] = 255

          # Check if outside boundaries
          p_w = p_ij + p_i
          if p_w[0] < -0.5*AREA_W or p_w[0] > 0.5*AREA_W or p_w[1] < -0.5*AREA_W or p_w[1] > 0.5*AREA_W:
            img_obs[0, i, j] = 255
          
          # Check for obstacles
          if p_w[0] > obstacles[0] - 0.6*OBS_W and p_w[0] < obstacles[0] + 0.6*OBS_W and p_w[1] > obstacles[1] - 0.6*OBS_W and p_w[1] < obstacles[1] + 0.6*OBS_W:
          # if np.linalg.norm(p_w - obs) < SAFETY_DIST:
            img_obs[0, i, j] = 255 

      # -------- EVAL -------
      region_id = vor.point_region[idx]
      cell = vor.regions[region_id]
      verts = vor.vertices[cell]
      poly = Polygon(verts)
      th = np.arange(0, 2*np.pi+np.pi/20, np.pi/20)
      rng_pts = p_i + ROBOT_RANGE * np.array([np.cos(th), np.sin(th)]).transpose()
      range_poly = Polygon(rng_pts)
      lim_region = intersection(poly, range_poly)

      dA = r_step**2
      for x_i in np.arange(p_i[0]-ROBOT_RANGE, p_i[0]+ROBOT_RANGE, r_step):
        for y_i in np.arange(p_i[1]-ROBOT_RANGE, p_i[1]+ROBOT_RANGE, r_step):
          pt = Point(x_i, y_i)
          if lim_region.contains(pt):
            dA_pdf = dA * gmm_pdf(x_i, y_i, means, covariances, mix)
            num += dA_pdf

      # Check collisions
      dist = np.linalg.norm(local_pts, axis=1)
      if (dist < 0.1).any():
        collision_counter[episode, s-1] = 1
        print("Collision detected with neighbors!")
      if p_i[0] > obstacles[0] - 0.5*OBS_W and p_i[0] < obstacles[0] + 0.5*OBS_W and p_i[1] > obstacles[1] - 0.5*OBS_W and p_i[1] < obstacles[1] + 0.5*OBS_W:
      # if np.linalg.norm(obs - p_i) < 0.5*SAFETY_DIST:
        collision_counter[episode, s-1] = 1
        print("Collision detected with obstacle!")
        failed = True
        

      img_i = np.concatenate((img_i, img_neighs), 0)
      img_i = np.concatenate((img_i, img_obs), 0)
      # print("img_i shape: ", img_i.shape)

      img_in = torch.from_numpy(img_i).unsqueeze(0)#.unsqueeze(0)
      img_in = img_in.to(torch.float).to(device)
      vel_i = model(img_in) * resolution
      vel_i = vel_i.squeeze(0)
      vel_i = vel_i.cpu().detach().numpy()

      # CBF
      if USE_CBF:
        local_pts = neighs - p_i
        constraints = []
        for n in local_pts:
          h = np.linalg.norm(n)**2 - SAFETY_DIST**2
          A_cbf = 2*n
          b_cbf = GAMMA * h
          constraints.append({'type': 'ineq', 'fun': lambda u: safety_constraint(u, A_cbf, b_cbf)})
        
        local_obs = obstacles - p_i
        for obs in local_obs:
          h = np.linalg.norm(obs)**2 - (2*SAFETY_DIST)**2
          A_cbf = 2*obs
          b_cbf = GAMMA * h
          constraints.append({'type': 'ineq', 'fun': lambda u: safety_constraint(u, A_cbf, b_cbf)})
        # print("vdes: ", vel_i)
        # print("Acbf: ", A_cbf)
        # print("b_cbf: ", b_cbf)
        # print("h: ", h)
        obj = lambda u: objective_function(u-vel_i)
        res = minimize(obj, vel_i, constraints=constraints, bounds=[(-vmax, vmax), (-vmax, vmax)])
        vel_i = res.x

      # print(f"Velocity of robot {idx}: {vel_i}")
      # print("points[idx] shape: ", points[idx, :].shape)
      points[idx, 0] = points[idx, 0] + vel_i[0]*dt
      points[idx, 1] = points[idx, 1] + vel_i[1]*dt

      if np.linalg.norm(vel_i) > 0.15:
        all_stopped = False
    
    robots_hist = np.concatenate((robots_hist, np.expand_dims(points, 0)))
    eta = num / denom
    # print("Efficiency: ", eta)
    eval_data[episode, s-1] = eta[0]

    

    if all_stopped:
      break
    
  path = Path().resolve()
  res_path = path / "results"
  
  if SAVE_POS:
    np.save(res_path/"eta16.npy", eval_data)
    np.save(res_path/"collisions16.npy", collision_counter)

  if not failed:
    np.save(res_path/"elle_pos.npy", robots_hist)


for i in range(ROBOTS_NUM):
  ax.plot(robots_hist[:, i, 0], robots_hist[:, i, 1])
  ax.scatter(robots_hist[-1, i, 0], robots_hist[-1, i, 1])


plt.show()



