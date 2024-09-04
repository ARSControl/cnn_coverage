import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import random
from shapely import Polygon, Point, intersection
from tqdm import tqdm
from pathlib import Path
import math
from scipy.optimize import minimize

from copy import deepcopy as dc
from sklearn.mixture import GaussianMixture

ROBOTS_NUM = 6
ROBOT_RANGE = 5.0
TARGETS_NUM = 2
COMPONENTS_NUM = 1
PARTICLES_NUM = 500
AREA_W = 20.0
vmax = 1.5
SAFETY_DIST = 2.0
EPISODES = 100
NUM_OBSTACLES = 4
USE_CBF = True
GAMMA = 0.5

path = Path().resolve()
path = (path / 'dataset_3ch_obs/')


def plot_occgrid(x, y, z, save=False, name="occgrid", ax=None):
  """
  Plot heatmap of occupancy grid.
  x, y, z : meshgrid
  """
  if save:
    path = Path("/unimore_home/mcatellani/pf-training/pics/")

  if ax is None:
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
  z_min = -1.0; z_max = 1.0
  c = ax.pcolormesh(x, y, z, cmap="YlOrRd", vmin=z_min, vmax=z_max)
  ax.set_xticks([]); ax.set_yticks([])
  if save:
    save_path = path / "{}.png".format(name)
    plt.savefig(str(save_path))
  if ax is None:
    plt.show()


def mirror(points):
    mirrored_points = []

    # Define the corners of the square
    square_corners = [(-0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, 0.5*AREA_W), (-0.5*AREA_W, 0.5*AREA_W)]

    # Mirror points across each edge of the square
    for edge_start, edge_end in zip(square_corners, square_corners[1:] + [square_corners[0]]):
        edge_vector = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])

        for point in points:
            # Calculate the vector from the edge start to the point
            point_vector = (point[0] - edge_start[0], point[1] - edge_start[1])

            # Calculate the mirrored point by reflecting across the edge
            mirrored_vector = (point_vector[0] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[0],
                               point_vector[1] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[1])

            # Translate the mirrored vector back to the absolute coordinates
            mirrored_point = (edge_start[0] + mirrored_vector[0], edge_start[1] + mirrored_vector[1])

            # Add the mirrored point to the result list
            mirrored_points.append(mirrored_point)

    return mirrored_points


def gauss_pdf(x, y, mean, covariance):

  points = np.column_stack([x.flatten(), y.flatten()])
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)

  return prob

def gmm_pdf(x, y, means, covariances, weights):
  prob = 0.0
  s = len(means)
  for i in range(s):
    prob += weights[i] * gauss_pdf(x, y, means[i], covariances[i])

  return prob

def objective_function(u):
  return np.linalg.norm(u)**2

def safety_constraint(u, A, b):
  return -np.dot(A,u) + b



for episode in range(EPISODES):
  targets = np.zeros((TARGETS_NUM, 1, 2))
  for i in range(TARGETS_NUM):
    targets[i, 0, 0] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand()
    targets[i, 0, 1] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand()

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
  gmm = GaussianMixture(n_components=TARGETS_NUM, covariance_type='full', max_iter=1000)
  gmm.fit(samples)

  means = gmm.means_
  covariances = gmm.covariances_
  mix = gmm.weights_

  print(f"Means: {means}")
  print(f"Covs: {covariances}")
  print(f"Mix: {mix}")


  ## Generate probability grid
  NUM_CHANNELS = 3
  GRID_STEPS = 64
  s = AREA_W/GRID_STEPS     # step

  xg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
  yg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
  Xg, Yg = np.meshgrid(xg, yg)
  Xg.shape

  Z = gmm_pdf(Xg, Yg, means, covariances, mix)
  Z = Z.reshape(GRID_STEPS, GRID_STEPS)
  Zmax = np.max(Z)
  Z = Z / Zmax

  fig, [ax, ax2] = plt.subplots(1, 2, figsize=(12,6))
  plot_occgrid(Xg, Yg, Z, ax=ax)
  plot_occgrid(Xg, Yg, Z, ax=ax2)
      
  


  # Simulate episode
  # ROBOTS_NUM = np.random.randint(6, ROBOTS_MAX)
  ROBOTS_NUM = 8
  converged = False
  NUM_STEPS = 50
  points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, 2)
  robots_hist = np.zeros((1, points.shape[0], points.shape[1]))
  robots_hist[0, :, :] = points
  vis_regions = []
  discretize_precision = 0.5

  imgs = np.zeros((1, ROBOTS_NUM, NUM_CHANNELS, GRID_STEPS, GRID_STEPS))
  vels = np.zeros((1, ROBOTS_NUM, 2))


  # obstacles
  obstacles = np.zeros((NUM_OBSTACLES, 2))
  for i in range(NUM_OBSTACLES):
    done = False
    while not done:
      obstacles[i, :] = -0.5*AREA_W + AREA_W*np.random.rand(1, 2)
      obs_rel = points - obstacles[i]
      norm = np.linalg.norm(obs_rel, axis=1)
      if (norm > 3.0).all():
        done = True   

  r_step = 2 * ROBOT_RANGE / GRID_STEPS
  for s in range(1, NUM_STEPS+1):
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
    img_s = np.zeros((ROBOTS_NUM, NUM_CHANNELS, GRID_STEPS, GRID_STEPS))
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

     

      img_s[idx, 0, :, :] = Z_i * 255

      neighs = np.delete(vor.points[:ROBOTS_NUM], idx, 0)
      local_pts = neighs - p_i

      # Remove undetected neighbors
      undetected = []
      for i in range(local_pts.shape[0]):
        if local_pts[i, 0] < -ROBOT_RANGE or local_pts[i, 0] > ROBOT_RANGE or local_pts[i, 1] < -ROBOT_RANGE or local_pts[i, 1] > ROBOT_RANGE:
          undetected.append(i)

      local_pts = np.delete(local_pts, undetected, 0)

      img_obs = np.zeros((GRID_STEPS, GRID_STEPS))
      img_neighs = np.zeros((GRID_STEPS, GRID_STEPS))
      for i in range(GRID_STEPS):
        for j in range(GRID_STEPS):
          # jj = GRID_STEPS-1-j
          p_ij = np.array([-ROBOT_RANGE+j*r_step, -ROBOT_RANGE+i*r_step])
          # print(f"Point ({i},{j}): {p_ij}")
          for n in local_pts:
            if np.linalg.norm(n - p_ij) <= SAFETY_DIST:
              img_neighs[i, j] = 1.0

          # Check if outside boundaries
          p_w = p_ij + p_i
          if p_w[0] < -0.5*AREA_W or p_w[0] > 0.5*AREA_W or p_w[1] < -0.5*AREA_W or p_w[1] > 0.5*AREA_W:
            img_obs[i, j] = 1.0
          
          # check obstacles
          for obs in obstacles:
            if p_w[0] > obs[0]-1 and p_w[0] < obs[0]+1 and p_w[1] > obs[1]-1 and p_w[1] < obs[1]+1:
              img_obs[i, j] = 1.0
        
      img_s[idx, 1, :, :] = img_neighs * 255
      img_s[idx, 2, :, :] = img_obs * 255
      region = vor.point_region[idx]
      poly_vert = []
      for vert in vor.regions[region]:
        v = vor.vertices[vert]
        poly_vert.append(v)
        # plt.scatter(v[0], v[1], c='tab:red')

      poly = Polygon(poly_vert)
      x,y = poly.exterior.xy
      # plt.plot(x, y, c='tab:orange')
      # robot = np.array([-18.0, -12.0])
      robot = vor.points[idx]

      # plt.scatter(robot[0], robot[1])

      # Intersect with robot range
      step = 0.5
      range_pts = []
      for th in np.arange(0.0, 2*np.pi, step):
        xi = robot[0] + ROBOT_RANGE * np.cos(th)
        yi = robot[1] + ROBOT_RANGE * np.sin(th)
        pt = Point(xi, yi)
        range_pts.append(pt)
        # plt.plot(xi, yi, c='tab:blue')

      range_poly = Polygon(range_pts)
      xc, yc = range_poly.exterior.xy

      lim_region = intersection(poly, range_poly)
      lim_regions.append(lim_region)

      # Calculate centroid with gaussian distribution
      xmin, ymin, xmax, ymax = lim_region.bounds
      # print(f"x range: {xmin} - {xmax}")
      # print(f"y range: {ymin} - {ymax}")
      A = 0.0
      Cx = 0.0; Cy = 0.0
      dA = discretize_precision ** 2
      # pts = [Point(xmin, ymin), Point(xmax, ymin), Point(xmax, ymax), Point(xmin, ymax)]
      # bound = Polygon(pts)
      for i in np.arange(xmin, xmax, discretize_precision):
        for j in np.arange(ymin, ymax, discretize_precision):
          pt_i = Point(i,j)
          if lim_region.contains(pt_i):
            dA_pdf = dA * gmm_pdf(i, j, means, covariances, mix)
            # print(dA_pdf)
            A = A + dA_pdf
            Cx += i*dA_pdf
            Cy += j*dA_pdf

      Cx = Cx / A
      Cy = Cy / A



      # centr = np.array([lim_region.centroid.x, lim_region.centroid.y])
      centr = np.array([Cx, Cy]).transpose()
      # print(f"Robot: {robot}")
      # print(f"Centroid: {centr}")
      dist = np.linalg.norm(robot-centr)
      vel = 0.8 * (centr - robot)
      vel[0, 0] = max(-vmax, min(vmax, vel[0,0]))
      vel[0, 1] = max(-vmax, min(vmax, vel[0,1]))
      vel_i = vel[0]
      
      # CBF
      if USE_CBF:
        local_pts = neighs - p_i
        constraints = []
        # for n in local_pts:
        #   h = np.linalg.norm(n)**2 - SAFETY_DIST**2
        #   A_cbf = 2*n
        #   b_cbf = GAMMA * h
        #   constraints.append({'type': 'ineq', 'fun': lambda u: safety_constraint(u, A_cbf, b_cbf)})
        
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
        vel = res.x
      
      vel_s[idx, :] = vel



      points[idx, :] = robot + vel
      if dist > 0.1:
        conv = False
    
    imgs = np.concatenate((imgs, np.expand_dims(img_s, 0)))
    vels = np.concatenate((vels, np.expand_dims(vel_s, 0)))

    # Save positions for visualization
    if s == 1:
      vis_regions.append(lim_regions)
    robots_hist = np.vstack((robots_hist, np.expand_dims(points, axis=0)))
    vis_regions.append(lim_regions)

    if conv:
      print(f"Episode {episode} Converged in {s} iterations")
      break
    # axs[row, s-1-5*row].scatter(points[:, 0], points[:, 1])

  imgs = imgs[1:]
  vels = vels[1:]

  """
  for i in range(ROBOTS_NUM):
    ax.plot(robots_hist[:, i, 0], robots_hist[:, i, 1])
    ax.scatter(robots_hist[-1, i, 0], robots_hist[-1, i, 1])

  for obs in obstacles:
    o_x = np.array([obs[0]-1, obs[0]+1, obs[0]+1, obs[0]-1, obs[0]-1])
    o_y = np.array([obs[1]-1, obs[1]-1, obs[1]+1, obs[1]+1, obs[1]-1])
    ax.plot(o_x, o_y, c='r', linewidth=3)

  for i in range(ROBOTS_NUM):
    ax2.scatter(robots_hist[-1, i, 0], robots_hist[-1, i, 1])

  plt.show()
  """
  
  with open(str(path/f"test{episode}.npy"), 'wb') as f:
    np.save(f, imgs)
  with open(str(path/f"vels{episode}.npy"), 'wb') as f:
    np.save(f, vels)
  
