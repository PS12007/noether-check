"""Reference solvers (pseudo-spectral, periodic, batched on the GPU) and their invariants.

KdV          u_t + 6 u u_x + u_xxx = 0           on [0, L), L = 40, N = 128
             conserves mass  M = int u,  momentum  P = int u^2,  energy  H = int (u_x^2 / 2 - u^3)
             time stepping: ETDRK4 (Kassam & Trefethen 2005), 2/3 dealiasing.

NS2D         w_t + (u . grad) w = nu lap w        on [0, 2 pi)^2, 64 x 64 (vorticity form)
             mean vorticity is conserved; kinetic energy E and enstrophy Z obey exact budgets
             dE/dt = -2 nu Z,  dZ/dt = -2 nu P (palinstrophy). Time stepping: RK4 with an
             integrating factor for the viscous term, 2/3 dealiasing.
"""
import math
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- KdV
class KdV:
    L, N = 40.0, 128

    def __init__(self, dt=0.002, device=DEV):
        self.dt, self.dev = dt, device
        N, L = self.N, self.L
        self.dx = L / N
        k = 2 * math.pi * torch.fft.rfftfreq(N, d=L / N).to(device, torch.float64)
        self.k = k
        self.dealias = (k < (2 / 3) * k.max()).to(torch.float64)
        Lin = 1j * k ** 3                                  # -u_xxx in Fourier space
        h = dt
        E, E2 = torch.exp(h * Lin), torch.exp(h * Lin / 2)
        # contour-integral evaluation of the ETDRK4 coefficients
        M = 32
        r = torch.exp(2j * math.pi * (torch.arange(1, M + 1, device=device, dtype=torch.float64) - 0.5) / M)   # full circle: L is imaginary
        LR = h * Lin[:, None] + r[None, :]
        self.Q = h * torch.mean((torch.exp(LR / 2) - 1) / LR, 1)
        self.f1 = h * torch.mean((-4 - LR + torch.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, 1)
        self.f2 = h * torch.mean((2 + LR + torch.exp(LR) * (-2 + LR)) / LR ** 3, 1)
        self.f3 = h * torch.mean((-4 - 3 * LR - LR ** 2 + torch.exp(LR) * (4 - LR)) / LR ** 3, 1)
        self.E, self.E2 = E, E2
        self.g = -3j * k * self.dealias                     # N(u) = -3 (u^2)_x

    def _N(self, vh):
        u = torch.fft.irfft(vh, n=self.N)
        return self.g * torch.fft.rfft(u * u)

    def step(self, vh):
        Nv = self._N(vh)
        a = self.E2 * vh + self.Q * Nv
        Na = self._N(a)
        b = self.E2 * vh + self.Q * Na
        Nb = self._N(b)
        c = self.E2 * a + self.Q * (2 * Nb - Nv)
        Nc = self._N(c)
        return self.E * vh + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3

    @torch.no_grad()
    def simulate(self, u0, t_save, n_save):
        """u0: [B, N] tensor. Returns [B, n_save + 1, N] float32 snapshots every t_save."""
        vh = torch.fft.rfft(u0.to(self.dev, torch.float64))
        per = int(round(t_save / self.dt))
        out = [u0.float().cpu()]
        for _ in range(n_save):
            for _ in range(per):
                vh = self.step(vh)
            out.append(torch.fft.irfft(vh, n=self.N).float().cpu())
        return torch.stack(out, 1)

    @staticmethod
    def x():
        return torch.arange(KdV.N, dtype=torch.float64) * KdV.L / KdV.N

    @staticmethod
    def solitons(B, rng, n_range=(1, 4), c_range=(0.3, 1.5)):
        """Sums of well-separated KdV solitons u = c/2 sech^2(sqrt(c)/2 (x - x0)), periodised."""
        x = KdV.x()
        u = torch.zeros(B, KdV.N, dtype=torch.float64)
        for b in range(B):
            n = rng.integers(n_range[0], n_range[1] + 1)
            x0 = rng.permutation(np.linspace(0, KdV.L, 8, endpoint=False))[:n] + rng.uniform(0, KdV.L / 8, n)
            for j in range(n):
                c = rng.uniform(*c_range)
                for shift in (-KdV.L, 0, KdV.L):
                    u[b] += c / 2 / torch.cosh(math.sqrt(c) / 2 * (x - x0[j] + shift)) ** 2
        return u

    @staticmethod
    def random_field(B, rng, amp=1.0, kmax=4):
        """Smooth random fields (out of distribution: they shed solitons and radiation)."""
        x = KdV.x()
        u = torch.zeros(B, KdV.N, dtype=torch.float64)
        for b in range(B):
            for k in range(1, kmax + 1):
                a, ph = rng.normal() / k, rng.uniform(0, 2 * math.pi)
                u[b] += a * torch.cos(2 * math.pi * k * x / KdV.L + torch.tensor(ph))
            u[b] = amp * u[b] / u[b].abs().max() * rng.uniform(0.5, 1.0)
        return u


def kdv_invariants(u):
    """u: [..., N] -> dict of [...] tensors: mass, momentum, energy (Hamiltonian)."""
    u = u.double()
    dx = KdV.L / KdV.N
    k = 2 * math.pi * torch.fft.rfftfreq(KdV.N, d=dx).to(u.device, torch.float64)
    ux = torch.fft.irfft(1j * k * torch.fft.rfft(u), n=KdV.N)
    return dict(mass=u.sum(-1) * dx, momentum=(u * u).sum(-1) * dx,
                energy=(0.5 * ux * ux - u ** 3).sum(-1) * dx)


# ----------------------------------------------------------------------------- 2D Navier-Stokes
class NS2D:
    N = 64
    L = 2 * math.pi

    def __init__(self, nu=1e-3, dt=0.01, device=DEV):
        self.nu, self.dt, self.dev = nu, dt, device
        N = self.N
        k = torch.fft.fftfreq(N, d=1.0 / N).to(device, torch.float64)
        kr = torch.fft.rfftfreq(N, d=1.0 / N).to(device, torch.float64)
        self.kx, self.ky = torch.meshgrid(k, kr, indexing="ij")   # x along dim -2, y along dim -1
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2inv = torch.where(self.k2 > 0, 1 / self.k2, torch.zeros_like(self.k2))
        kmax = N // 2
        self.dealias = ((self.kx.abs() < 2 / 3 * kmax) & (self.ky.abs() < 2 / 3 * kmax)).to(torch.float64)

    def _rhs(self, wh):
        psih = wh * self.k2inv
        u = torch.fft.irfft2(1j * self.ky * psih, s=(self.N, self.N))     # u = psi_y
        v = torch.fft.irfft2(-1j * self.kx * psih, s=(self.N, self.N))    # v = -psi_x
        wx = torch.fft.irfft2(1j * self.kx * wh, s=(self.N, self.N))
        wy = torch.fft.irfft2(1j * self.ky * wh, s=(self.N, self.N))
        return -torch.fft.rfft2(u * wx + v * wy) * self.dealias

    def step(self, wh):
        h = self.dt
        E = torch.exp(-self.nu * self.k2 * h / 2)
        k1 = self._rhs(wh)
        k2 = self._rhs(E * (wh + h / 2 * k1))
        k3 = self._rhs(E * wh + h / 2 * k2)
        k4 = self._rhs(E * E * wh + h * E * k3)
        return E * E * wh + h / 6 * (E * E * k1 + 2 * E * (k2 + k3) + k4)

    @torch.no_grad()
    def simulate(self, w0, t_save, n_save):
        wh = torch.fft.rfft2(w0.to(self.dev, torch.float64))
        per = int(round(t_save / self.dt))
        out = [w0.float().cpu()]
        for _ in range(n_save):
            for _ in range(per):
                wh = self.step(wh)
            out.append(torch.fft.irfft2(wh, s=(self.N, self.N)).float().cpu())
        return torch.stack(out, 1)

    @staticmethod
    def random_field(B, rng, k0=4.0, amp=1.0):
        """Random vorticity with energy spectrum peaked near wavenumber k0, zero mean."""
        N = NS2D.N
        k = np.fft.fftfreq(N, 1.0 / N)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        kk = np.sqrt(kx ** 2 + ky ** 2)
        spec = kk ** 4 * np.exp(-2 * (kk / k0) ** 2)     # vorticity amplitude spectrum
        out = np.empty((B, N, N))
        for b in range(B):
            ph = np.exp(2j * np.pi * rng.random((N, N)))
            w = np.real(np.fft.ifft2(np.sqrt(spec) * ph))
            w -= w.mean()
            out[b] = amp * w / w.std()
        return torch.tensor(out)


def ns_invariants(w, nu=1e-3):
    """w: [..., N, N] vorticity -> mean vorticity, kinetic energy, enstrophy, palinstrophy
    (all as domain means, so they are resolution independent)."""
    w = w.double()
    N = w.shape[-1]
    k = torch.fft.fftfreq(N, d=1.0 / N).to(w.device, torch.float64)
    kr = torch.fft.rfftfreq(N, d=1.0 / N).to(w.device, torch.float64)
    kx, ky = torch.meshgrid(k, kr, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    k2inv = torch.where(k2 > 0, 1 / k2, torch.zeros_like(k2))
    wh = torch.fft.rfft2(w)
    psih = wh * k2inv
    u = torch.fft.irfft2(1j * ky * psih, s=(N, N))
    v = torch.fft.irfft2(-1j * kx * psih, s=(N, N))
    wx = torch.fft.irfft2(1j * kx * wh, s=(N, N))
    wy = torch.fft.irfft2(1j * ky * wh, s=(N, N))
    m = lambda f: f.mean((-2, -1))
    return dict(mean=m(w), energy=0.5 * m(u * u + v * v), enstrophy=0.5 * m(w * w),
                palinstrophy=0.5 * m(wx * wx + wy * wy))
