import torch


def randomQ(shape=(), device=None, dtype=None):
    """
    Haar-random unit quaternions, ordered (..., z, w, x, y).

    shape gives any leading batch dimensions.
    randomQ()       -> (4,)
    randomQ((B,))   -> (B, 4)
    randomQ((B,L))  -> (B, L, 4)
    """
    if isinstance(shape, int):
        shape = (shape,)

    q = torch.randn(*shape, 4, device=device, dtype=dtype)
    return q / torch.linalg.norm(q, dim=-1, keepdim=True)

def rotFromQ(q):
    """
    SO(3) rotation matrices from unit quaternions q = (..., z, w, x, y).

    q : (..., 4)
    R : (..., 3, 3)
    """
    z, w, x, y = q.unbind(dim=-1)

    col0 = torch.stack((
        1 - 2*(y*y + z*z),
        2*(x*y + w*z),
        2*(x*z - w*y)
    ), dim=-1)

    col1 = torch.stack((
        2*(x*y - w*z),
        1 - 2*(x*x + z*z),
        2*(y*z + w*x)
    ), dim=-1)

    col2 = torch.stack((
        2*(x*z + w*y),
        2*(y*z - w*x),
        1 - 2*(x*x + y*y)
    ), dim=-1)

    return torch.stack((col0, col1, col2), dim=-1)

def vFromQ(q):
    """
    Image of the z-axis under q.

    q : (..., 4)
    v : (..., 3)
    """
    z, w, x, y = q.unbind(dim=-1)

    return torch.stack((
        2*(x*z + w*y),
        2*(y*z - w*x),
        1 - 2*(x*x + y*y)
    ), dim=-1)


def qFromV(v):
    """Random U(1) fiber representative above unit vector v (..., 3)."""
    x0, y0, z0 = v.unbind(dim=-1)

    phi = 2 * torch.pi * torch.rand(
        v.shape[:-1], device=v.device, dtype=v.dtype
    )
    c, s = torch.cos(phi), torch.sin(phi)

    north = z0 >= 0
    south = ~north

    q = torch.empty((*v.shape[:-1], 4), device=v.device, dtype=v.dtype)

    # North patch
    zp = torch.sqrt((1 + z0[north]) / 2)
    xp = x0[north] / (2 * zp)
    yp = y0[north] / (2 * zp)

    q[north] = torch.stack((
        c[north] * zp,
        s[north] * zp,
        c[north] * xp - s[north] * yp,
        s[north] * xp + c[north] * yp
    ), dim=-1)

    # South patch
    xpp = torch.sqrt((1 - z0[south]) / 2)
    zpp = x0[south] / (2 * xpp)
    wpp = -y0[south] / (2 * xpp)

    q[south] = torch.stack((
        c[south] * zpp - s[south] * wpp,
        s[south] * zpp + c[south] * wpp,
        c[south] * xpp,
        s[south] * xpp
    ), dim=-1)

    return q