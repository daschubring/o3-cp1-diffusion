import numpy as np
from scipy.special import spherical_in

import sphere


def randomSpins(nCfg, L):
    """Generate nCfg independent random spin configurations on S^2."""
    spins = np.random.normal(size=(nCfg, L, 3))
    return sphere.normalize(spins)


def heatBathStep(m, beta):
    """
    Sample new O(3) spins in the local field m.

    The conditional distribution is von Mises-Fisher with mean direction
    m/|m| and concentration kappa = beta |m|.
    """
    m = np.asarray(m, dtype=float)
    mNorm = sphere.norm(m)
    kappa = beta * mNorm

    # The direction of m is irrelevant when |m| = 0 because kappa = 0
    # and sphere.vMF samples uniformly. Give those entries an arbitrary
    # valid unit direction so that we never divide by zero.
    zeroField = mNorm[..., 0] < 1e-12
    mUnit = np.empty_like(m)
    np.divide(m, mNorm, out=mUnit, where=mNorm > 0)
    mUnit[zeroField] = sphere.zVec

    return sphere.vMF(mUnit, kappa)


def heatBathSweep(spins, beta):
    """
    Perform one checkerboard heat-bath sweep of a periodic 1D O(3) lattice.

    spins may have any leading batch dimensions, with shape (..., L, 3).
    The lattice length L must be even. The array is updated in place and
    also returned.
    """
    spins = np.asarray(spins)

    if spins.shape[-1] != 3:
        raise ValueError("spins must have shape (..., L, 3)")

    L = spins.shape[-2]
    if L % 2 != 0:
        raise ValueError("checkerboard heat-bath updates require even L")

    # Update even sites using their two odd neighbors.
    spinsOdd = spins[..., 1::2, :]
    evenNeighbor = spinsOdd + np.roll(spinsOdd, 1, axis=-2)
    spins[..., ::2, :] = heatBathStep(evenNeighbor, beta)

    # Update odd sites using the newly updated even neighbors.
    spinsEven = spins[..., ::2, :]
    oddNeighbor = spinsEven + np.roll(spinsEven, -1, axis=-2)
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

    Returns an array of length L containing C(r) for r = 0, ..., L-1.
    Averages over all lattice sites and all leading batch dimensions.
    """
    cfgs = np.asarray(cfgs)
    L = cfgs.shape[-2]
    corr = np.empty(L, dtype=float)

    for r in range(L):
        products = sphere.dot(np.roll(cfgs, -r, axis=-2), cfgs)
        corr[r] = np.mean(products)

    return corr


def exactCorrelation(L, beta, ellMax=50):
    """Exact finite-volume correlation function for the periodic 1D O(3) model."""
    ellAll = np.arange(ellMax + 2)

    # Transfer-matrix eigenvalues are proportional to i_ell(beta).
    # Dividing by i_0 improves numerical conditioning; the common factor
    # cancels from the correlation function.
    lam = spherical_in(ellAll, beta)
    lam = lam / lam[0]

    ell = np.arange(ellMax + 1)
    Z = np.sum((2 * ell + 1) * lam[ell]**L)

    corr = np.empty(L, dtype=float)
    for r in range(L):
        numerator = np.sum(
            (ell + 1)
            * (
                lam[ell]**(L - r) * lam[ell + 1]**r
                + lam[ell + 1]**(L - r) * lam[ell]**r
            )
        )
        corr[r] = numerator / Z

    return corr


def energy(cfgs):
    """
    Energy H = -sum_i s_i . s_{i+1} for each configuration (J = 1).
    """
    cfgs = np.asarray(cfgs)
    bonds = sphere.dot(cfgs, np.roll(cfgs, -1, axis=-2))
    return -np.sum(bonds, axis=-2)[..., 0]


def energyVariance(cfgs):
    """Ensemble variance of H, using the population variance (ddof=0)."""
    return np.var(energy(cfgs))


def exactEnergyVariance(L, beta, ellMax=50):
    """Exact finite-volume variance of the energy for the periodic 1D O(3) model."""
    ell = np.arange(ellMax + 1)

    # Need i_l through l = ellMax + 1.
    i = spherical_in(np.arange(ellMax + 2), beta)
    iL = i[:-1]
    iLp1 = i[1:]

    # First and second beta derivatives of i_l(beta).
    A = iLp1 + (ell / beta) * iL
    B = (
        (1 + ell * (ell - 1) / beta**2) * iL
        - (2 / beta) * iLp1
    )

    # Common rescaling for numerical stability. It cancels from all ratios.
    scale = iL[0]
    iL = iL / scale
    A = A / scale
    B = B / scale

    degeneracy = 2 * ell + 1
    Z = np.sum(degeneracy * iL**L)

    Zprime = L * np.sum(
        degeneracy * iL**(L - 1) * A
    )

    Zdoubleprime = L * np.sum(
        degeneracy
        * (
            (L - 1) * iL**(L - 2) * A**2
            + iL**(L - 1) * B
        )
    )

    return Zdoubleprime / Z - (Zprime / Z)**2
