import torch

import sphere


def randomSpins(nCfg, L, device=None, dtype=torch.float32):
    """Generate nCfg independent random spin configurations on S^2."""
    spins = torch.randn((nCfg, L, 3), device=device, dtype=dtype)
    return sphere.normalize(spins)


def heatBathStep(m, beta):
    """
    Sample new O(3) spins in the local field m.

    The conditional distribution is von Mises-Fisher with mean direction
    m/|m| and concentration kappa = beta |m|.
    """
    if m.shape[-1] != 3:
        raise ValueError("m must have shape (..., 3)")

    mNorm = sphere.norm(m)
    kappa = beta * mNorm

    # For zero local field the direction is irrelevant because kappa = 0.
    # Give those entries an arbitrary unit direction to avoid division by zero.
    zeroField = mNorm < 1e-12
    mUnit = m / torch.clamp(mNorm, min=1e-12)
    zVec = m.new_tensor([0.0, 0.0, 1.0])
    mUnit = torch.where(zeroField, zVec, mUnit)

    return sphere.vMF(mUnit, kappa)


def heatBathSweep(spins, beta):
    """
    Perform one checkerboard heat-bath sweep of a periodic 1D O(3) lattice.

    spins may have any leading batch dimensions, with shape (..., L, 3).
    The lattice length L must be even. The tensor is updated in place and
    also returned.
    """
    if spins.shape[-1] != 3:
        raise ValueError("spins must have shape (..., L, 3)")

    L = spins.shape[-2]
    if L % 2 != 0:
        raise ValueError("checkerboard heat-bath updates require even L")

    # Update even sites using their two odd neighbors.
    spinsOdd = spins[..., 1::2, :]
    evenNeighbor = spinsOdd + torch.roll(spinsOdd, 1, dims=-2)
    spins[..., ::2, :] = heatBathStep(evenNeighbor, beta)

    # Update odd sites using the newly updated even neighbors.
    spinsEven = spins[..., ::2, :]
    oddNeighbor = spinsEven + torch.roll(spinsEven, -1, dims=-2)
    spins[..., 1::2, :] = heatBathStep(oddNeighbor, beta)

    return spins


def heatBath(spins, beta, nSweeps=1):
    """Perform nSweeps checkerboard heat-bath sweeps in place."""
    for _ in range(nSweeps):
        heatBathSweep(spins, beta)
    return spins


def correlation(cfgs):
    """
    Translationally averaged spin correlation C(r) = <s_i . s_{i+r}>.

    Parameters
    ----------
    cfgs : torch.Tensor
        Shape (..., L, 3). All leading dimensions and lattice sites are
        averaged over.

    Returns
    -------
    torch.Tensor
        Shape (L,), containing C(r) for r = 0, ..., L-1. The result remains
        on the same device as cfgs.
    """
    if cfgs.shape[-1] != 3:
        raise ValueError("cfgs must have shape (..., L, 3)")

    L = cfgs.shape[-2]
    corr = torch.empty(L, device=cfgs.device, dtype=cfgs.dtype)

    for r in range(L):
        products = sphere.dot(torch.roll(cfgs, -r, dims=-2), cfgs)
        corr[r] = torch.mean(products)

    return corr
