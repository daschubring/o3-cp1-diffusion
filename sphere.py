import numpy as np
from scipy.special import eval_legendre, i0e, i1e

#global unit vectors
zVec = np.array([0., 0., 1.])
yVec = np.array([0., 1., 0.])

def norm(v):
    return np.linalg.norm(v, axis=-1, keepdims=True)

def normalize(v):
    return v/norm(v)

def dot(v,w):
    out = np.sum(v * w, axis=-1)
    return out

def hkScalar(mUnit, t):
    """
    Sample heat kernel at time t centered on unit vector mUnit
    Uses rejection sampling compared to a von Mises-Fisher distribution
    Uses scalar quantities. Not suitable for batched inputs
    """

    kappa = 1/(2*t)

    while True:
        nProp = vMF(mUnit, kappa)
        cosTh = dot(nProp, mUnit)

        acceptProb = (
            hkDist(cosTh, t)
            / (rejection(t)*vMFDist(cosTh, kappa))
        )

        if np.random.random() < acceptProb:
            return nProp

def hk(mUnit, t):
    """
    Sample the S^2 heat kernel for a batch of mean directions.
    """

    mUnit = np.asarray(mUnit)
    samples = np.empty_like(mUnit)

    kappa = 1/(2*t)
    remaining = np.arange(len(mUnit))

    while len(remaining) > 0:

        means = mUnit[remaining]
        kappaBatch = np.full((len(remaining), 1), kappa)

        nProp = vMF(means, kappaBatch)
        cosTh = dot(nProp, means).ravel()

        acceptProb = (
            hkDist(cosTh, t)
            / (rejection(t)*vMFDist(cosTh, kappa))
        )

        accepted = np.random.random(len(remaining)) < acceptProb

        samples[remaining[accepted]] = nProp[accepted]
        remaining = remaining[~accepted]

    return samples

def vMF(mUnit, kappa):
    """
    Sample von Mises-Fisher distribution about a batch of mean unit vectors mUnit (should work for a single vector too), with concentration parameter kappa
    """

    # Convert Python lists/scalars to numpy arrays.
    # WARNING: for batched use this function assumes mUnit has shape (...,3)
    # and kappa has shape (...,1), so that scalar quantities broadcast over vectors.
    mUnit = np.asarray(mUnit, dtype=float)
    kappa = np.asarray(kappa, dtype=float)

    # WARNING: mUnit is assumed to contain unit vectors and kappa is assumed nonnegative.

    # mask that locates small kappa
    smallKappa = np.abs(kappa) < 1e-8
    regular = ~smallKappa
    
    #sample random variables that will be used to construct the new spin
    chi = np.random.random(kappa.shape)
    phi = 2*np.pi*np.random.random(kappa.shape)

    #Consider the equilibrium distribution for cosTh, the dot product of v and mUnit
    cosTh = np.empty_like(kappa, dtype=float)

    # For kappa -> 0, vMF becomes uniform on the sphere, so cosTh is uniform on [-1,1]
    cosTh[smallKappa] = 2*chi[smallKappa] - 1

    # Algebraically equivalent to the inverse-CDF expression, but unlike expressions
    # containing exp(kappa), this remains well behaved for very large kappa.
    # exp(-2*kappa) may underflow to zero for large kappa; this is harmless.
    cosTh[regular] = 1 + (1/kappa[regular])*np.log(
        chi[regular] + (1-chi[regular])*np.exp(-2*kappa[regular])
    )

    cosTh = np.clip(cosTh,-1,1)
    sinTh = np.sqrt(1 - cosTh**2)

    w = np.cross(zVec, mUnit)
    
    # If mUnit is parallel to z, use y instead as the reference axis
    # WARNING: this assumes norm(w) retains a final singleton axis, e.g. shape (...,1).
    wNorm = norm(w)
    parallelToZ = wNorm[..., 0] < 1e-12

    if np.any(parallelToZ):
       w[parallelToZ] = np.cross(yVec, mUnit[parallelToZ])
         
    wUnit = normalize(w)
    xUnit = np.cross(mUnit,wUnit)

    return cosTh*mUnit + sinTh*(np.cos(phi)*wUnit + np.sin(phi)*xUnit)

#normalized von Mises-Fisher density on S^2 as a function of cos(theta)
def vMFDist(cosTh, kappa):

    cosTh = np.asarray(cosTh, dtype=float)
    kappa = np.asarray(kappa, dtype=float)

    smallKappa = np.abs(kappa) < 1e-8
    regular = ~smallKappa

    dist = np.empty(np.broadcast_shapes(cosTh.shape, kappa.shape), dtype=float)
    cosTh, kappa = np.broadcast_arrays(cosTh, kappa)

    # kappa -> 0 gives the uniform distribution on S^2
    dist[smallKappa] = 1/(4*np.pi)

    # Same as kappa*exp(kappa*cosTh)/(4*pi*sinh(kappa)),
    # rearranged to avoid overflow at large kappa.
    dist[regular] = (
        kappa[regular]
        * np.exp(kappa[regular]*(cosTh[regular] - 1))
        / (2*np.pi*(-np.expm1(-2*kappa[regular])))
    )

    return dist

# numerically optimized rejection-envelope values for 0.1 < t < 1.2
rejectionT = np.arange(0.1, 1.21, 0.05)
rejectionM = np.array([1.042925, 1.064141, 1.081439, 1.092474, 1.097214, 1.096852, 1.092859, 1.086554, 1.079008, 1.071053, 1.063342, 1.056405, 1.050692, 1.046612, 1.044556, 1.044920, 1.048123, 1.054626, 1.064952, 1.079711, 1.099626, 1.122410, 1.142114], dtype=float)

rejectionSafety = 0.01

def rejection(t):

    t = np.asarray(t, dtype=float)

    if t < 0.15:
        return 1 + 0.5*t

    elif t > 1.2:
        return (
            t*np.expm1(1/t)
            * (1 - 3*np.exp(-2*t) + 5*np.exp(-6*t))
        )

    else:
        return np.interp(t, rejectionT, rejectionM) + rejectionSafety * ((1.2-t)/1.1 + .1*(t-0.1)/1.1)
        # I'm linearly decreasing rejectionSafety so that rejection(t) is closer to continuous at 1.2

#the heat kernel distribution function, which selects from a number of approximations depending on cosTh and t
def hkDist(cosTh, t):
    """
    Practical S^2 heat-kernel density used for sampling.
    """

    cosTh = np.asarray(cosTh, dtype=float)
    th = np.arccos(np.clip(cosTh, -1, 1))

    # spectral representation for moderate and large diffusion times
    if t >= 0.15:
        return hkSpectralAdaptive(cosTh, t)

    # short-time MPSD representation
    dist = hkMPSD(th, t)

    # only check the boundary-layer cutoff in the tiny antipodal region
    thCheck = 2.90569
    checkBL = th > thCheck

    if np.any(checkBL):
        useBL = checkBL & (th > thBL(t))
        dist = np.where(useBL, hkBL(th, t), dist)

    return dist

#Individual approximations used in hkDist:

def hkSpectral(cosTh, t, lMax):
    """S^2 heat kernel from the spherical-harmonic expansion."""
    
    cosTh = np.asarray(cosTh)
    hk = np.zeros_like(cosTh, dtype=float)

    for ell in range(lMax + 1):
        hk += (
            (2 * ell + 1)
            * eval_legendre(ell, cosTh)
            * np.exp(-ell * (ell + 1) * t)
        )

    return hk / (4 * np.pi)

# minimum t for which each lMax is sufficiently accurate
# (so lMax=0 is good enough for t > 6.2, lMax=1 is for 6.2 > t > 2.2 etc.)
spectralTMin = [6.2, 2.2, 1.1, 0.65, 0.44, 0.32, 0.24, 0.19, 0.15, 0.12, 0.1]
def lMaxValue(t):
    for lMax in range(len(spectralTMin)):
        if t >= spectralTMin[lMax]:
            return lMax

    raise ValueError(
        f"Spectral approximation not calibrated below t={spectralTMin[-1]}"
    )

def hkSpectralAdaptive(cosTh, t):
    """
    Spectral S^2 heat kernel with lMax chosen from calibrated
    accuracy thresholds.
    """

    return hkSpectral(cosTh, t, lMaxValue(t))

def hkMPSD(th, t):
    """
    Short-time MPSD approximation on S^2, including corrections
    through O(t^4) with theta powers counted as theta^2 ~ t.
    """

    # Note it is sufficient up to the ~10^{-6} integrated error we use throughout to use
    # O(t^4) for 0.10 < t < 0.15
    # O(t^3) for 0.05 < t < 0.10
    # O(t^2) for 0.02 < t < 0.05
    # O(t) for t < 0.02
    # But since O(t^4) is inexpensive I use it for all t < 0.15

    th = np.asarray(th, dtype=float)

    th2 = th**2

    correction = (
        1
        + (
            1/12
            + th2/180
            + th2**2/1890
            + th2**3/18900
        )*t
        + (
            7/480
            + 13*th2/5040
            + 19*th2**2/50400
        )*t**2
        + (
            31/8064
            + 157*th2/120960
        )*t**3
        + 127*t**4/92160
    )

    return hkMPSDleading(th, t)*correction

def hkMPSDleading(th, t):
    """Leading order Minakshisundaram-Pleijel-Schwinger-DeWitt (MPSD) approximation to the heat kernel.
        Asymptotic corrections are handled directly in hkDist."""
    
    th = np.asarray(th)

    # np.sinc(th/pi) = sin(th)/th, and is well behaved at th = 0
    vanVleck = 1 / np.sqrt(np.sinc(th / np.pi))

    return (
        vanVleck
        * np.exp(-th**2 / (4 * t) + t / 4)
        / (4 * np.pi * t)
    )


def hkPi(t, nMin=-1, nMax=0):
    """Camporesi image sum evaluated exactly at theta = pi,
       truncated to nMin <= n <= nMax.
    """
    
    imageSum = 0.0

    for n in range(nMin, nMax + 1):
        sign = 1 if n % 2 == 0 else -1

        imageSum += (
            sign
            * (2 * n + 1)
            * np.exp(
                -np.pi**2 * (2 * n + 1)**2 / (4 * t)
            )
        )

    prefactor = (
        np.pi**2
        * np.exp(t / 4)
        / (4 * np.pi * t)**1.5
    )

    return prefactor * imageSum


thBLT = np.arange(0.01, 0.22, 0.01)
thBLVals = np.array([3.137, 3.13239, 3.12775, 3.12309, 3.02291, 3.00816, 2.99434, 2.98132, 2.96898, 2.95723, 2.94603, 2.93531, 2.92504, 2.91517, 2.90569, 2.89656, 2.88777, 2.87929, 2.87112, 2.86323, 2.85561], dtype=float)
    
def thBL(t):
    """
    Boundary between the MPSD and antipodal boundary-layer regions.
    Calibrated for 0.01 <= t <= 0.21.
    """

    if t < thBLT[0]:
        return thBLVals[0]

    elif t > thBLT[-1]:
        raise ValueError("thBL is only defined for the short-time regime t <= 0.21")

    else:
        return np.interp(t, thBLT, thBLVals)

def hkBL(th, t, method="zero"):
    """
    Handle the antipodal boundary-layer region of the S^2 heat kernel.

    method:
        "zero"       : discard the exponentially suppressed region
        "pi"         : replace by the (near) exact value of the kernel at theta = pi
        "asymptotic" : use the boundary-layer asymptotic expansion
    """

    if method == "zero":
        return np.zeros_like(th, dtype=float)

    elif method == "pi":
        return np.full_like(th, hkPi(t), dtype=float)

    elif method == "asymptotic":
        return hkBLasymptotic(th, t)

    else:
        raise ValueError(f"Unknown boundary-layer method: {method}")

from scipy.special import i0e, i1e

def hkBLasymptotic(th, t):
    """
    Boundary-layer asymptotic approximation near theta = pi,
    including the first subleading correction.
    """

    th = np.asarray(th, dtype=float)

    delta = np.pi - th
    z = np.pi*delta/(2*t)

    prefactor = (
        np.sqrt(np.pi)/(4*t**1.5)
        * np.exp(t/4 - np.pi**2/(4*t) + np.abs(z))
    )

    leading = i0e(z)

    correction = (
        t/np.pi**2
        * (z**2*i0e(z) + z*i1e(z))
    )

    return prefactor*(leading - correction)

#Uglier but more efficient batched version of hk (for use with numpy). Unused in first implementation
def hkBatched(mUnit, t):

    mUnit = np.asarray(mUnit, dtype=float)

    samples = np.empty_like(mUnit)
    pending = np.ones(mUnit.shape[:-1], dtype=bool)

    kappa = 1/(2*t)

    while np.any(pending):

        kappaBatch = np.full(pending.sum(), kappa)[..., None]

        nProp = vMF(mUnit[pending], kappaBatch)
        cosTh = dot(nProp, mUnit[pending])

        acceptProb = (
            hkDist(cosTh, t)
            / (rejection(t)*vMFDist(cosTh, kappaBatch))
        )

        accepted = np.random.random(acceptProb.shape) < acceptProb
        accepted = accepted[..., 0]

        samplesPending = samples[pending]
        samplesPending[accepted] = nProp[accepted]
        samples[pending] = samplesPending

        pendingPending = pending[pending]
        pendingPending[accepted] = False
        pending[pending] = pendingPending

    return samples