import matplotlib.pyplot as plt
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

from torchesnufft.functional import get_density, nufft1, nufft2, nufft_inv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode="reflect", channel_axis=None) + 0.0j
image = torch.from_numpy(image).to(device).unsqueeze(0).unsqueeze(0)

# Set up radial sampling trajectory
num_spokes = 600
num_samples_per_spoke = 320
M = num_spokes * num_samples_per_spoke
N1, N2 = image.shape[-2:]

# Generate 2D radial trajectory in k-space
r = torch.linspace(-torch.pi, torch.pi, num_samples_per_spoke + 1, device=device)[:-1]
line_x = r
line_y = torch.zeros_like(r)
angles = torch.linspace(0, torch.pi, num_spokes + 1, device=device)[:-1]
x_list, y_list = list(), list()
for angle in angles:
    x_rot = line_x * torch.cos(angle) - line_y * torch.sin(angle)
    y_rot = line_x * torch.sin(angle) + line_y * torch.cos(angle)
    x_list.append(x_rot)
    y_list.append(y_rot)
x = torch.cat(x_list)
y = torch.cat(y_list)
xy = torch.stack((x, y))

# Plot the trajectory
plt.figure()
plt.plot(x.cpu(), y.cpu(), "o", markersize=3)
plt.xlabel("kx")
plt.ylabel("ky")
plt.title("2D radial trajectory")
plt.axis("equal")
plt.show()
plt.close()

# Forward NUFFT
c = nufft2(-xy, image)

# Density compensation: weight by distance to center
ram_lak = torch.linalg.norm(xy, dim=0)
reco_analytic = nufft1(xy, c * ram_lak[None, None, ...], (N1, N2))

# Inverse NUFFT
density = get_density(xy, c, (N1, N2))
reco = nufft_inv(xy, c, (N1, N2))

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
for spoke in range(num_spokes):
    axs[0].plot(ram_lak.cpu().reshape(num_spokes, num_samples_per_spoke)[spoke])
    axs[0].set_title("Ram-Lak filter")
    axs[1].plot(density.cpu().reshape(num_spokes, num_samples_per_spoke)[spoke])
    axs[1].set_title("Density torchesnufft")
plt.show()
plt.close()

# Plot the trajectory colored by ram_lak and density
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sc1 = ax1.scatter(xy[0].cpu(), xy[1].cpu(), c=ram_lak.cpu(), cmap="viridis", s=8)
ax1.set_title("Trajectory colored by Ram-Lak")
ax1.set_xlabel("kx")
ax1.set_ylabel("ky")
cbar1 = fig.colorbar(sc1, ax=ax1)
cbar1.set_label("Density")
sc2 = ax2.scatter(xy[0].cpu(), xy[1].cpu(), c=density.cpu(), cmap="viridis", s=8)
ax2.set_title("Trajectory colored by torchesnufft density")
ax2.set_xlabel("kx")
ax2.set_ylabel("ky")
cbar2 = fig.colorbar(sc2, ax=ax2)
cbar2.set_label("Density")

plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.imshow(image[0, 0].cpu().abs(), cmap="gray")
plt.title("Shepp-Logan")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(reco_analytic[0, 0].cpu().abs(), cmap="gray")
plt.title("Reconstruction (Analytic)")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(reco[0, 0].cpu().abs(), cmap="gray")
plt.title("Reconstruction")
plt.axis("off")
plt.show()
plt.close()
