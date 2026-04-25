import matplotlib.pyplot as plt
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

from torchesnufft.functional import get_density, nufft1, nufft2, nufft_inv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode="reflect", channel_axis=None) + 0.0j
image = torch.from_numpy(image).to(device).unsqueeze(0).unsqueeze(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
# Standard uniform FFT
f_uniform = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(image)))
ax1.set_title("Uniform FFT")
ax1.imshow(torch.log(torch.abs(f_uniform.squeeze()) + 1e-8).cpu(), cmap=plt.cm.Greys_r)
# Uniform torchesnufft (forward)
N1, N2 = image.shape[-2:]
x = -2 * torch.pi * torch.arange(-N1 // 2, N1 // 2) / N1
y = -2 * torch.pi * torch.arange(-N2 // 2, N2 // 2) / N2
xy = torch.reshape(torch.stack(torch.meshgrid(x, y, indexing="ij")), (2, -1))
f_nufft = nufft2(xy.to(device), image[None, None, ...]).reshape(N1, N2)
ax2.set_title("Uniform torchesnufft type 2")
ax2.imshow(torch.log(torch.abs(f_nufft).cpu().squeeze() + 1e-8), cmap=plt.cm.Greys_r)
plt.show()
plt.close()
print("MSE torchesnufft nufft2:", torch.mean(torch.abs(f_nufft - f_uniform) ** 2).item())


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5))
ax1.set_title("Ground-truth")
ax1.imshow(torch.abs(image.squeeze().cpu()), cmap=plt.cm.Greys_r)
# Standard uniform iFFT
reco_uniform = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(f_uniform)))
ax2.set_title("Uniform iFFT")
ax2.imshow(torch.abs(reco_uniform.squeeze().cpu()), cmap=plt.cm.Greys_r)
# Uniform torchesnufft (adjoint)
N1, N2 = image.shape[-2:]
x = 2 * torch.pi * torch.arange(-N1 // 2, N1 // 2) / N1
y = 2 * torch.pi * torch.arange(-N2 // 2, N2 // 2) / N2
xy = torch.reshape(torch.stack(torch.meshgrid(x, y, indexing="ij")), (2, -1))
reco_nufft = nufft1(xy.to(device), f_uniform.flatten()[None, None], (N1, N2)) / (N1 * N2)
ax3.set_title("Uniform torchesnufft type 1")
ax3.imshow(torch.abs(reco_nufft.cpu().squeeze()), cmap=plt.cm.Greys_r)
plt.show()
plt.close()

print("MSE torchesnufft nufft1:", torch.mean(torch.abs(reco_nufft - reco_uniform) ** 2).item())


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
# Analytic density
density_uniform = torch.abs(torch.ones_like(f_uniform))
ax1.set_title("Analytic density")
ax1.imshow(density_uniform.squeeze().cpu(), cmap=plt.cm.Greys_r)
# Density torchesnufft
N1, N2 = image.shape[-2:]
x = 2 * torch.pi * torch.arange(-N1 // 2, N1 // 2) / N1
y = 2 * torch.pi * torch.arange(-N2 // 2, N2 // 2) / N2
xy = torch.reshape(torch.stack(torch.meshgrid(x, y, indexing="ij")), (2, -1))
density_nufft = get_density(xy.to(device), f_uniform.flatten()[None, None], (N1, N2))
ax2.set_title("Density torchesnufft")
ax2.imshow(density_nufft.squeeze().cpu().reshape(N1, N2), cmap=plt.cm.Greys_r)
plt.show()
plt.close()

print(
    "MSE torchesnufft density:",
    torch.mean(torch.abs(density_nufft.reshape(N1, N2) - density_uniform) ** 2).item(),
)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5))
ax1.set_title("Ground-truth")
ax1.imshow(torch.abs(image.squeeze().cpu()), cmap=plt.cm.Greys_r)
# Standard uniform iFFT
reco_uniform = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(f_uniform)))
ax2.set_title("Uniform iFFT")
ax2.imshow(torch.abs(reco_uniform.squeeze().cpu()), cmap=plt.cm.Greys_r)
# Uniform torchesnufft (inverse)
N1, N2 = image.shape[-2:]
x = 2 * torch.pi * torch.arange(-N1 // 2, N1 // 2) / N1
y = 2 * torch.pi * torch.arange(-N2 // 2, N2 // 2) / N2
xy = torch.reshape(torch.stack(torch.meshgrid(x, y, indexing="ij")), (2, -1))
reco_nufft = nufft_inv(xy.to(device), f_uniform.flatten()[None, None], (N1, N2)) / (N1 * N2)
ax3.set_title("Uniform torchesnufft inverse")
ax3.imshow(torch.abs(reco_nufft.cpu().squeeze()), cmap=plt.cm.Greys_r)
plt.show()
plt.close()

print("MSE torchesnufft inverse:", torch.mean(torch.abs(reco_nufft - reco_uniform) ** 2).item())
