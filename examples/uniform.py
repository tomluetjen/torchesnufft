import matplotlib.pyplot as plt
import torch
import torchkbnufft
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale

from torchesnufft.functional import nufft1, nufft2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode="reflect", channel_axis=None) + 0.0j
image = torch.from_numpy(image).to(device).unsqueeze(0).unsqueeze(0)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5))


# Standard uniform FFT
f_uniform = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(image)))
ax1.set_title("uniform FFT")
ax1.imshow(torch.log(torch.abs(f_uniform.squeeze()) + 1e-8).cpu(), cmap=plt.cm.Greys_r)


# Uniform torchesnufft
N1, N2 = image.shape[-2:]
x = -2 * torch.pi * torch.arange(-N1 // 2, N1 // 2) / N1
y = -2 * torch.pi * torch.arange(-N2 // 2, N2 // 2) / N2
xy = torch.reshape(torch.stack(torch.meshgrid(x, y, indexing="ij")), (2, -1))
image_flat = torch.reshape(image, (1, 1, -1))
f_nufft = nufft1(xy.to(device), image_flat, (N1, N2))

ax2.set_title("Uniform torchesnufft Type 1")
ax2.imshow(torch.log(torch.abs(f_nufft).cpu().squeeze() + 1e-8), cmap=plt.cm.Greys_r)

# Uniform torchkbnufft
kb_ob = torchkbnufft.KbNufft(im_size=image.shape[-2:], device=device)
f_kbnufft_flat = kb_ob(
    image,
    -xy.to(device).unsqueeze(0),
)
f_kbnufft = torch.reshape(f_kbnufft_flat, (N1, N2))

ax3.set_title("Uniform torchkbnufft Type 1")
ax3.imshow(
    torch.log(torch.abs(f_kbnufft).cpu().squeeze() + 1e-8),
    cmap=plt.cm.Greys_r,
)

plt.show()
plt.close()


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4.5))

ax1.set_title("Original image")
ax1.imshow(torch.abs(image.squeeze().cpu()), cmap=plt.cm.Greys_r)

# Standard uniform iFFT
reco_uniform = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(f_uniform)))
ax2.set_title("Uniform iFFT")
ax2.imshow(torch.abs(reco_uniform.squeeze().cpu()), cmap=plt.cm.Greys_r)

# Uniform torchesnufft Type 2
N1, N2 = image.shape[-2:]
x = 2 * torch.pi * torch.arange(-N1 // 2, N1 // 2) / N1
y = 2 * torch.pi * torch.arange(-N2 // 2, N2 // 2) / N2
xy = torch.reshape(torch.stack(torch.meshgrid(x, y, indexing="ij")), (2, -1))

reco_nufft = torch.reshape(nufft2(xy.to(device), f_uniform), (N1, N2)) / (N1 * N2)

ax3.set_title("Uniform torchesnufft Type 2")
ax3.imshow(torch.abs(reco_nufft.cpu().squeeze()), cmap=plt.cm.Greys_r)

# Uniform torchkbnufft Type 2
kb_ob = torchkbnufft.KbNufftAdjoint(im_size=image.shape[-2:], device=device)
f_uniform_flat = torch.reshape(f_uniform, (1, 1, -1))
reco_kbnufft = kb_ob(
    f_uniform_flat,
    xy.to(device).unsqueeze(0),
) / (N1 * N2)

ax4.set_title("Uniform torchkbnufft Type 2")
ax4.imshow(
    torch.abs(reco_kbnufft).cpu().squeeze(),
    cmap=plt.cm.Greys_r,
)

plt.show()
plt.close()
