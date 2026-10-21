"""Neural surrogates. Each maps the state at time t to the state at t + 0.5 (residual form:
u_next = u + scale * net(u / scale)). All layers are periodic and translation equivariant,
like the equations. Works in 1D ([B, N]) and 2D ([B, N, N])."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(nd):
    return nn.Conv1d if nd == 1 else nn.Conv2d


class SpectralConv(nn.Module):
    def __init__(self, c, modes, nd):
        super().__init__()
        self.nd, self.modes = nd, modes
        scale = 1 / (c * c)
        shape = (c, c, modes) if nd == 1 else (c, c, 2 * modes, modes)
        self.w = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    def forward(self, x):
        m = self.modes
        if self.nd == 1:
            xh = torch.fft.rfft(x.float())
            out = torch.zeros_like(xh)
            out[..., :m] = torch.einsum("bix,iox->box", xh[..., :m], self.w)
            return torch.fft.irfft(out, n=x.shape[-1])
        xh = torch.fft.rfft2(x.float())
        out = torch.zeros_like(xh)
        idx = torch.cat([torch.arange(m), torch.arange(xh.shape[-2] - m, xh.shape[-2])]).to(x.device)
        out[..., idx, :m] = torch.einsum("bixy,ioxy->boxy", xh[..., idx, :m], self.w)
        return torch.fft.irfft2(out, s=x.shape[-2:])


class FNO(nn.Module):
    def __init__(self, nd=1, width=64, modes=16, layers=4):
        super().__init__()
        C = _conv(nd)
        self.nd = nd
        self.lift = C(1, width, 1)
        self.spec = nn.ModuleList(SpectralConv(width, modes, nd) for _ in range(layers))
        self.pw = nn.ModuleList(C(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(C(width, 128, 1), nn.GELU(), C(128, 1, 1))

    def forward(self, u):
        h = self.lift(u.unsqueeze(1))
        for i, (s, p) in enumerate(zip(self.spec, self.pw)):
            h = s(h) + p(h)
            if i < len(self.spec) - 1:
                h = F.gelu(h)
        return self.proj(h).squeeze(1)


class ResBlock(nn.Module):
    def __init__(self, c, nd, k, dil):
        super().__init__()
        C = _conv(nd)
        self.c1 = C(c, c, k, padding=dil * (k // 2), dilation=dil, padding_mode="circular")
        self.c2 = C(c, c, k, padding=k // 2, padding_mode="circular")

    def forward(self, h):
        return h + self.c2(F.gelu(self.c1(F.gelu(h))))


class ResNet(nn.Module):
    def __init__(self, nd=1, width=64, blocks=6, k=5):
        super().__init__()
        C = _conv(nd)
        self.inp = C(1, width, k, padding=k // 2, padding_mode="circular")
        self.blocks = nn.Sequential(*[ResBlock(width, nd, k, 2 ** (i % 3)) for i in range(blocks)])
        self.out = C(width, 1, k, padding=k // 2, padding_mode="circular")

    def forward(self, u):
        return self.out(F.gelu(self.blocks(self.inp(u.unsqueeze(1))))).squeeze(1)


class UNet(nn.Module):
    def __init__(self, nd=1, width=32, depth=3, k=3):
        super().__init__()
        C = _conv(nd)
        conv = lambda a, b: nn.Sequential(C(a, b, k, padding=k // 2, padding_mode="circular"), nn.GELU(),
                                          C(b, b, k, padding=k // 2, padding_mode="circular"), nn.GELU())
        self.nd = nd
        ch = [width * 2 ** i for i in range(depth + 1)]
        self.inp = conv(1, ch[0])
        self.down = nn.ModuleList(conv(ch[i], ch[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(conv(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(depth)))
        self.out = C(ch[0], 1, 1)

    def forward(self, u):
        pool = F.avg_pool1d if self.nd == 1 else F.avg_pool2d
        h = self.inp(u.unsqueeze(1))
        skips = [h]
        for d in self.down:
            h = d(pool(h, 2))
            skips.append(h)
        skips.pop()
        for up in self.up:
            h = F.interpolate(h, scale_factor=2, mode="linear" if self.nd == 1 else "bilinear", align_corners=False)
            h = up(torch.cat([h, skips.pop()], 1))
        return self.out(h).squeeze(1)


ARCHS = dict(fno=FNO, resnet=ResNet, unet=UNet)


class Surrogate(nn.Module):
    """Residual wrapper with fixed input / output scales."""

    def __init__(self, arch, nd, scale_in, scale_out, **kw):
        super().__init__()
        self.net = ARCHS[arch](nd=nd, **kw)
        self.register_buffer("s_in", torch.tensor(float(scale_in)))
        self.register_buffer("s_out", torch.tensor(float(scale_out)))

    def forward(self, u):
        return u + self.s_out * self.net(u / self.s_in)
