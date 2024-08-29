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

import torch
from torch import nn


from utils import *
from models import *

import pycrazyswarm

np.random.seed(0)

ROBOTS_NUM = 2
ROBOT_RANGE = 0.5
TARGETS_NUM = 1
COMPONENTS_NUM = 1
PARTICLES_NUM = 500
AREA_W =3.0
vmax = 1.5
SAFETY_DIST = .2
EPISODES = 1
NUM_STEPS = 100
MU = [0.0, 0.0]
sleepRate = 1000
k_position = 0.5
des_z = 0.5

num = 0.0

eval_data = np.zeros((EPISODES, NUM_STEPS))

path = Path().resolve()
path = path / "trained_models"
MODEL_PATH = path / '2d_cnn.pt'
print("Model Path ", MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model = SimpleCNN().to(device)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print(model)

for episode in range(EPISODES):
    targets = np.zeros((TARGETS_NUM, 1, 2))
    for i in range(TARGETS_NUM):
        targets[i, 0, 0] = -0.5 * (AREA_W - 1) + (AREA_W - 1) * np.random.rand(1, 1)
        targets[i, 0, 1] = -0.5 * (AREA_W - 1) + (AREA_W - 1) * np.random.rand(1, 1)

    STD_DEV = 0.5
    samples = np.zeros((TARGETS_NUM, PARTICLES_NUM, 2))
    for k in range(TARGETS_NUM):
        for i in range(PARTICLES_NUM):
            samples[k, i, :] = targets[k, 0, :] + STD_DEV * np.random.randn(1, 2)

    # Fit GMM
    samples = samples.reshape((TARGETS_NUM * PARTICLES_NUM, 2))
    print(samples.shape)
    gmm = GaussianMixture(n_components=COMPONENTS_NUM, covariance_type='full', max_iter=1000)
    gmm.fit(samples)

    means = [MU]
    covariances = gmm.covariances_
    mix = gmm.weights_

    print(f"Means: {means}")
    print(f"Covs: {covariances}")
    print(f"Mix: {mix}")

    ## -------- Generate decentralized probability grid ---------
    GRID_STEPS = 64
    s = AREA_W / GRID_STEPS  # step

    xg = np.linspace(-0.5 * AREA_W, 0.5 * AREA_W, GRID_STEPS)
    yg = np.linspace(-0.5 * AREA_W, 0.5 * AREA_W, GRID_STEPS)
    Xg, Yg = np.meshgrid(xg, yg)
    Xg.shape
    print(Xg.shape)

    Z = gmm_pdf(Xg, Yg, means, covariances, mix)
    Z = Z.reshape(GRID_STEPS, GRID_STEPS)
    Zmax = np.max(Z)
    Z = Z / Zmax

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_aspect('equal')
    plot_occgrid(Xg, Yg, Z, ax=ax)

    # ---------- Simulate episode ---------
    converged = False
    vis_regions = []
    discretize_precision = 0.5
    imgs = np.zeros((1, ROBOTS_NUM, GRID_STEPS, GRID_STEPS))
    vels = np.zeros((1, ROBOTS_NUM, 2))
    vx_tot=[]
    vy_tot=[]

    swarm = pycrazyswarm.Crazyswarm()
    timeHelper = swarm.timeHelper
    cfs = swarm.allcfs.crazyflies

    init_pos = np.empty((ROBOTS_NUM, 3))
    for i, cf in enumerate(cfs):
        print(f"CF {cf.id} -> Initial position: {cf.position()}")
        init_pos[i] = cf.position()

    # Check for initial position
    for i, cf in enumerate(cfs):
        if cf.position()[0] <= -1.5 or cf.position()[0] >= 1.5 or cf.position()[1] <= -1.5 or cf.position()[1] >= 1.5:
            print(f"CF-{cf.id} out of the region!")
            break

    # Robot History
    robots_hist = []
    for i, cf in enumerate(cfs):
        robots_hist.append(np.empty((0, 2)))
        robots_hist[i] = np.vstack((robots_hist[i], np.array([[cf.position()[0], cf.position()[1]]])))

    points = np.empty((ROBOTS_NUM, 2)) # Robot positions

    for i, cf in enumerate(cfs):
        points[i] = cf.position()[:2]

    print(points)


    swarm.allcfs.takeoff(targetHeight=des_z, duration=3.0)
    timeHelper.sleep(5)  

    r_step = 2 * ROBOT_RANGE / GRID_STEPS
    denom = np.sum(s**2 * gmm_pdf(Xg, Yg, means, covariances, mix))
    print("Total info: ", denom)

    """
    #############
    # MAIN LOOP #
    ############# 
    """
    for s in range(1, NUM_STEPS + 1):

        points = np.array([cf.position()[:2] for cf in cfs])

        plt.cla()
        plot_occgrid(Xg, Yg, Z, ax=ax)

        vx_tot=[]
        vy_tot=[]
        print(f"*** Step {s} ***")
        row = 0
        if s > 5:
            row = 1

        # all_stopped = True

        # mirror points across each edge of the env
        dummy_points = np.zeros((5 * ROBOTS_NUM, 2))
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
            Z_i = gmm_pdf(Xg_i, Yg_i, means - p_i, covariances, mix)
            Z_i = Z_i.reshape(GRID_STEPS, GRID_STEPS)
            Zmax_i = np.max(Z_i)
            Z_i = Z_i / Zmax_i

            neighs = np.delete(vor.points[:ROBOTS_NUM], idx, 0)
            local_pts = neighs - p_i

            # Remove undetected neighbors
            undetected = []
            for i in range(local_pts.shape[0]):
                if local_pts[i, 0] < -ROBOT_RANGE or local_pts[i, 0] > ROBOT_RANGE or local_pts[i, 1] < -ROBOT_RANGE or \
                        local_pts[i, 1] > ROBOT_RANGE:
                    undetected.append(i)

            local_pts = np.delete(local_pts, undetected, 0)

            img_i = dc(Z_i)
            for i in range(GRID_STEPS):
                for j in range(GRID_STEPS):
                    # jj = GRID_STEPS-1-j
                    p_ij = np.array([-ROBOT_RANGE + j * r_step, -ROBOT_RANGE + i * r_step])
                    # print(f"Point ({i},{j}): {p_ij}")
                    for n in local_pts:
                        if np.linalg.norm(n - p_ij) <= SAFETY_DIST:
                            img_i[i, j] = -1.0

                    # Check if outside boundaries
                    p_w = p_ij + p_i
                    if p_w[0] < -0.5 * AREA_W or p_w[0] > 0.5 * AREA_W or p_w[1] < -0.5 * AREA_W or p_w[
                        1] > 0.5 * AREA_W:
                        img_i[i, j] = -1.0

            img_in = torch.from_numpy(img_i).unsqueeze(0).unsqueeze(0)
            img_in = img_in.to(torch.float).to(device)
            vels_i = model(img_in)

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

            # CBF
            R = 0.2
            gamma_safe = 0.2
            vels_i = vels_i.cpu().detach().numpy()
            # Creare l'array numpy per u_des
            u_des = np.array([vels_i[0, 0], vels_i[0, 1]])

            def objective_function(u):
                return np.linalg.norm(u) ** 2

            # Vincolo di sicurezza
            def safety_constraint1(u, A_1, b_1):
                return - np.dot(A_1, u) + b_1

            A_1 = np.array([])
            b_1 = np.array([])
            constraints = []
            h = 0
            for i in range(ROBOTS_NUM):
                # considero tutti i robot tranne robot idx
                if i != idx:
                    h = np.linalg.norm(points[idx] - points[i]) ** 2 - R ** 2
                    A_1 = np.array([-2 * (points[idx] - points[i])])
                    b_1 = np.array([gamma_safe * h])

                    constraints.append({'type': 'ineq', 'fun': lambda u: safety_constraint1(u, A_1, b_1)})

            obj = lambda u: objective_function(u - u_des)  # voglio minimizzare la differenza tra v ottima e desiderata.

            result = minimize(obj, u_des, constraints=constraints, bounds=[(-0.15, 0.15), (-0.15, 0.15)])
            u_1 = result.x

            vx_tot.append(u_1[0])
            vy_tot.append(u_1[1])

        for idx, cf in enumerate(cfs):
            startTime = timeHelper.time()
            errorZ = des_z - cf.position()[2]
            vz = k_position * errorZ
            cf.cmdVelocityWorld(np.array([vx_tot[idx], vy_tot[idx], vz]), yawRate=0)
            print(f"Drone {cf.id} - vx: {vx_tot[idx]}, vy: {vy_tot[idx]}")
            timeHelper.sleepForRate(sleepRate)
        
        for i, cf in enumerate(cfs):
            robots_hist[i] = np.vstack((robots_hist[i], points[i]))

        eta = num / denom
        eval_data[episode, s-1] = eta

        for i in range(ROBOTS_NUM):
            ax.plot(robots_hist[i][:, 0], robots_hist[i][:, 1], c="k")
            ax.scatter(robots_hist[i][-1, 0], robots_hist[i][-1, 1], c="b")

        plt.pause(0.001)

landingPoses = []
for cf in cfs:
    landingPoses.append(np.array([cf.position()[0], cf.position()[1], 0.1]))
landed = []
while len(landed) != ROBOTS_NUM:
    for idx, cf in enumerate(cfs):
        if not cf.id in landed:
            if np.linalg.norm(cf.position() - landingPoses[idx]) < 0.1:
                cf.cmdStop()
                landed.append(cf.id)
                print(f"Drone {cf.id} landed!")
            else:
                cf.cmdPosition(landingPoses[idx], yaw=0)
    timeHelper.sleepForRate(sleepRate)

path = Path().resolve()
res_path = path / "results"
np.save(res_path/"eta.npy", eval_data)
np.save(res_path/"robots_hist.npy", robots_hist)

# fai vedere il grafico finale del percorso dei droni
for i in range(ROBOTS_NUM):
    ax.plot(robots_hist[i][:, 0], robots_hist[i][:, 1], c="k")
    ax.scatter(robots_hist[i][-1, 0], robots_hist[i][-1, 1], c="b")

plt.show()