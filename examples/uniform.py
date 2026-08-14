import matplotlib.pyplot as plt
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

from torchesnufft.functional import get_density, nufft1, nufft2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode="reflect", channel_axis=None) + 0.0j
image = torch.from_numpy(image).to(device).unsqueeze(0).unsqueeze(0)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5))
# Standard uniform FFT
f_uniform = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(image)))
ax1.set_title("Uniform FFT")
ax1.imshow(torch.log(torch.abs(f_uniform.squeeze()) + 1e-8).cpu(), cmap=plt.cm.Greys_r)
# Uniform torchesnufft
N1, N2 = image.shape[-2:]
x = 2 * torch.pi * torch.arange(-N1 // 2, N1 // 2) / N1
y = 2 * torch.pi * torch.arange(-N2 // 2, N2 // 2) / N2
xy = torch.reshape(torch.stack(torch.meshgrid(x, y, indexing="ij")), (2, -1))
f_nufft1 = nufft1(-xy.to(device), image.flatten()[None, None], (N1, N2))
ax2.set_title("Uniform type 1")
ax2.imshow(torch.log(torch.abs(f_nufft1).cpu().squeeze() + 1e-8), cmap=plt.cm.Greys_r)
c_nufft2 = nufft2(-xy.to(device), image[None, None, ...]).reshape(N1, N2)
ax3.set_title("Uniform type 2")
ax3.imshow(torch.log(torch.abs(c_nufft2).cpu().squeeze() + 1e-8), cmap=plt.cm.Greys_r)
plt.show()
plt.close()
print("MSE nufft1 (forward):", torch.mean(torch.abs(f_nufft1 - f_uniform) ** 2).item())
print("MSE nufft2 (forward):", torch.mean(torch.abs(c_nufft2 - f_uniform) ** 2).item())


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4.5))
ax1.set_title("Ground-truth")
ax1.imshow(torch.abs(image.squeeze().cpu()), cmap=plt.cm.Greys_r)
# Standard uniform iFFT
reco_uniform = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(f_uniform)))
ax2.set_title("Uniform iFFT")
ax2.imshow(torch.abs(reco_uniform.squeeze().cpu()), cmap=plt.cm.Greys_r)
# Uniform torchesnufft
reco_nufft1 = nufft1(xy.to(device), f_uniform.flatten()[None, None], (N1, N2)) / (N1 * N2)
ax3.set_title("Uniform type 1")
ax3.imshow(torch.abs(reco_nufft1.cpu().squeeze()), cmap=plt.cm.Greys_r)
reco_nufft2 = nufft2(xy.to(device), f_uniform[None, None]).reshape(N1, N2) / (N1 * N2)
ax4.set_title("Uniform type 2")
ax4.imshow(torch.abs(reco_nufft2.cpu().squeeze()), cmap=plt.cm.Greys_r)
plt.show()
plt.close()

print(
    "MSE nufft1 (adjoint):",
    torch.mean(torch.abs(reco_nufft1 - reco_uniform) ** 2).item(),
)
print(
    "MSE nufft2 (adjoint):",
    torch.mean(torch.abs(reco_nufft2 - reco_uniform) ** 2).item(),
)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
# Analytic density
density_uniform = torch.abs(torch.ones_like(f_uniform))
ax1.set_title("Analytic density")
ax1.imshow(density_uniform.squeeze().cpu(), cmap=plt.cm.Greys_r)
# Density torchesnufft
get_density = get_density(xy.to(device), f_uniform.flatten()[None, None], (N1, N2))
ax2.set_title("Estimated density")
ax2.imshow(get_density.squeeze().cpu().reshape(N1, N2), cmap=plt.cm.Greys_r)
plt.show()
plt.close()

print(
    "MSE get_density:",
    torch.mean(torch.abs(get_density.reshape(N1, N2) - density_uniform) ** 2).item(),
)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5))
ax1.set_title("Ground-truth")
ax1.imshow(torch.abs(image.squeeze().cpu()), cmap=plt.cm.Greys_r)
# Standard uniform iFFT
reco_uniform = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(f_uniform)))
ax2.set_title("Uniform iFFT")
ax2.imshow(torch.abs(reco_uniform.squeeze().cpu()), cmap=plt.cm.Greys_r)
# Uniform torchesnufft with density compensation
reco_nufft1 = nufft1(
    xy.to(device), f_uniform.flatten()[None, None] * get_density.flatten()[None, None], (N1, N2)
) / (N1 * N2)
ax3.set_title("Uniform torchesnufft inverse")
ax3.imshow(torch.abs(reco_nufft1.cpu().squeeze()), cmap=plt.cm.Greys_r)
plt.show()
plt.close()

print(
    "MSE nufft1 (inverse):",
    torch.mean(torch.abs(reco_nufft1 - reco_uniform) ** 2).item(),
)
