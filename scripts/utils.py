import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path




def plot_occgrid(x, y, z, save=False, name="occgrid", ax=None, colormap="YlOrRd"):
  """
  Plot heatmap of occupancy grid.
  x, y, z : meshgrid
  """
  if save:
    path = Path("/unimore_home/mcatellani/pf-training/pics/")

  if ax is None:
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
  z_min = 0.0; z_max = 1.0
  c = ax.pcolormesh(x, y, z, cmap=colormap, vmin=z_min, vmax=z_max)
  ax.set_xticks([]); ax.set_yticks([])
  if save:
    save_path = path / "{}.png".format(name)
    plt.savefig(str(save_path))
  if ax is None:
    plt.show()


def mirror(points, AREA_W=20.0):
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


def gauss3d_pdf(x, y, z, mean, covariance):

  points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)

  return prob


def gmm3d_pdf(x, y, z, means, covs, weights):
  n_comps = len(means)
  prob = 0.0
  for i in range(n_comps):
    prob += weights[i] * gauss3d_pdf(x, y, z, means[i], covs[i])
  
  return prob