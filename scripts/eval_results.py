import numpy as np
import math
from pathlib import Path
from matplotlib import pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help="plot eta")
    parser.add_argument('--save', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help="save plot")
    args = parser.parse_args()
    return args


args = parse_args()


path = Path.home() / "cnn_coverage/results"
files = [a for a in path.glob("**/*")]
print("Number of files: ", len(files))

ns = [12, 16, 20]
etas = []
collisions = []
etas_std = []
collisions_std = []
for n in ns:
    etas_std.append(np.load(path/f"eta{n}_std_4_obs.npy"))
    collisions_std.append(np.load(path/f"collisions{n}_std_4_obs.npy"))
    etas.append(np.load(path/f"eta{n}.npy"))
    collisions.append(np.load(path/f"collisions{n}.npy"))


for e in range(len(etas)):
    for i in range(etas[e].shape[0]):
        if etas[e][i, 0] == 0:
            break
    etas[e] = etas[e][:i+1, :]
   
print("Eta shape after removing: ", etas[0].shape)

for e in range(len(etas)):
    ids = (etas[e]==0).argmax(axis=1)
    for i in range(etas[e].shape[0]):
        etas[e][i, ids[i]:] = etas[e][i, ids[i]-1]


for e in range(len(etas_std)):
    for i in range(etas_std[e].shape[0]):
        if etas_std[e][i, 0] == 0:
            break
    etas_std[e] = etas_std[e][:i, :]
   
# print("Eta shape after removing: ", etas[0].shape)

for e in range(len(etas_std)):
    ids = (etas_std[e]==0).argmax(axis=1)
    for i in range(etas_std[e].shape[0]):
        etas_std[e][i, ids[i]:] = etas[e][i, ids[i]-1]

etas_m = []
stds = []
for eta in etas:
    etas_m.append(np.mean(eta, axis=0))
    stds.append(np.std(eta, axis=0))
    print("Our eta: ", etas_m[-1][-1])


etas_std_m = []
stds_s = []
for eta in etas_std:
    etas_std_m.append(np.mean(eta, axis=0))
    stds_s.append(np.std(eta, axis=0))
    print("std eta: ", etas_std_m[-1][-1])


collisions_num = []
for i in range(len(collisions)):
    tot = 0
    c = collisions[i]
    c[:, :20] = 0
    for j in range(c.shape[0]):
        a = np.where(c[j, :] == 1)
        if a[0].shape[0] > 0:
            tot += 1
    collisions_num.append(tot)
    print(f"OURS: Number of collisions with {ns[i]} robots: {tot}")


collisions_num = []
for i in range(len(collisions_std)):
    tot = 0
    c = collisions_std[i]
    c[:, :20] = 0
    for j in range(c.shape[0]):
        a = np.where(c[j, :] == 1)
        if a[0].shape[0] > 0:
            tot += 1
    collisions_num.append(tot)
    print(f"STD: Number of collisions with {ns[i]} robots: {tot}")
 

t = np.arange(etas[0].shape[0])
fig, [ax, ax2] = plt.subplots(1, 2, figsize=(12,6))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, eta in enumerate(etas_m):
    ax.plot(t, eta, label=f"{ns[i]} Robots", c=colors[i])
    ax.fill_between(t, eta-stds[i], eta+stds[i], color=colors[i], alpha=0.2)
ax.title.set_text("proposed effectiveness")

for i, eta in enumerate(etas_std_m):
    ax2.plot(t, eta, label=f"{ns[i]} Robots", c=colors[i])
    ax2.fill_between(t, eta-stds_s[i], eta+stds_s[i], color=colors[i], alpha=0.2)
ax2.title.set_text("traditional effectiveness")
ax.grid(); ax2.grid()
ax.legend(); ax2.legend()

if args.save:
    figpath = Path.home() / "cnn_coverage/pics"
    plt.savefig(figpath/"eta_comparison.png")
if args.plot:
    plt.show()
