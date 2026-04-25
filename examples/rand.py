import matplotlib.pyplot as plt
import torch

from torchesnufft.functional import nufft2, nufft_inv


def make_signal(n, device):
    t = torch.linspace(-1.0, 1.0, n, device=device)
    signal = torch.exp(-40.0 * (t) ** 2)
    return signal, t


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)

N = 50
M = 1000

# Generate gaussian signal and time stamps
signal, t = make_signal(N, device)
signal = signal.unsqueeze(0).unsqueeze(0)

# Random points in [-pi, pi]
x = torch.rand((1, M), device=device) * 2 * torch.pi - torch.pi

# Forward and inverse transforms.
c = nufft2(-x, signal)
reco = nufft_inv(x, c, (N,)) / N

mse = torch.mean(torch.abs(reco - signal) ** 2).item()
print(f"MSE: {mse:.2e}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

ax1.plot(t.cpu(), signal[0, 0].abs().cpu(), label="Ground-truth", linewidth=2)
ax1.plot(t.cpu(), reco[0, 0].abs().cpu(), "--", label="Reconstruction", linewidth=2)
ax1.set_xlabel("Time [a.u.]")
ax1.set_ylabel("Signal intensity [a.u.]")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(x[0].cpu(), c[0, 0].abs().cpu(), ".", markersize=3)
ax2.set_xlabel("Time [a.u.]")
ax2.set_ylabel("Signal intensity [a.u.]")
ax2.grid(alpha=0.3)

plt.show()
