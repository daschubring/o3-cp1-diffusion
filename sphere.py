import numpy as np
from scipy.special import eval_legendre, i0e, i1e, eval_jacobi

#global unit vectors
zVec = np.array([0., 0., 1.])
yVec = np.array([0., 1., 0.])

#the boundary between large and short times for heat kernel distribution and score
tThresh = 0.15
tThreshScore = 0.137

#default harmonic truncations
lDist = 8
lScore = 10

#theta above which special handling takes place at short times
thCheck = 2.90569
thCheckScore = 2.84627

def norm(v):
    return np.linalg.norm(v, axis=-1, keepdims=True)

def normalize(v):
    return v/norm(v)

def dot(v,w):
    out = np.sum(v * w, axis=-1, keepdims=True)
    return out

def tangent(x, x0):
    """
    Tangent vector at x pointing towards x0. Magnitude sin theta
    """
    cosTh = dot(x, x0)
    return x0 - cosTh*x

def randomPerp(mUnit):
    # trial perpendicular
    w = np.cross(zVec, mUnit)
    
    # If mUnit is parallel to z, use y instead as the reference axis
    wNorm = norm(w)
    parallelToZ = wNorm[..., 0] < 1e-12
    w[parallelToZ] = np.cross(yVec, mUnit[parallelToZ])

    # Create orthonormal basis
    wUnit = normalize(w)
    xUnit = np.cross(mUnit,wUnit)

    # Sample random angle
    phi = 2*np.pi*np.random.random(wNorm.shape)

    # Rotate wUnit
    return np.cos(phi)*wUnit + np.sin(phi)*xUnit 

def broadcastFix(vector, scalar):
    vectorShape = np.broadcast(vector, scalar).shape
    scalarShape = vectorShape[:-1] + (1,)

    vector = np.broadcast_to(vector, vectorShape)
    scalar = np.broadcast_to(scalar, scalarShape)

    return vector, scalar

def hk(mUnit, t):
    """
    Sample the S^2 heat kernel
    """

    mUnit, t = broadcastFix(mUnit, t)
    
    samples = np.empty_like(mUnit)

    kappa = 1/(2*t)

    #while loop acting on indices that have not had an accepted proposal
    remaining = np.arange(len(mUnit))
    while len(remaining) > 0:
        tLeft = t[remaining]
        kappaLeft = kappa[remaining]
        mLeft = mUnit[remaining]

        #propose a vector from vMF
        nProp = vMF(mLeft, kappaLeft)
        cosTh = dot(nProp, mLeft)

        acceptProb = (hkDist(cosTh, tLeft) / (rejection(tLeft)*vMFDist(cosTh, kappaLeft))).ravel()

        accepted = np.random.random(acceptProb.shape) < acceptProb

        samples[remaining[accepted]] = nProp[accepted]
        remaining = remaining[~accepted]

    return samples

def vMF(mUnit, kappa):
    """
    Sample von Mises-Fisher distribution about unit vector mUnit, with concentration parameter kappa
    """

    mUnit, kappa = broadcastFix(mUnit, kappa)

    # mask that locates small kappa
    smallKappa = np.abs(kappa) < 1e-8
    regular = ~smallKappa
    
    #random variable to construct cosTh
    chi = np.random.random(kappa.shape)
    cosTh = np.empty(kappa.shape, dtype=float)

    #for smallKappa, cosTh is uniform
    cosTh[smallKappa] = 2*chi[smallKappa] - 1

    #for regular, we use the vMF distribution
    cosTh[regular] = 1 + (1/kappa[regular])*np.log(chi[regular] + (1-chi[regular])*np.exp(-2*kappa[regular]))

    cosTh = np.clip(cosTh,-1,1)
    sinTh = np.sqrt(1 - cosTh**2)

    return cosTh*mUnit + sinTh*randomPerp(mUnit)

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
    mFactor = np.empty_like(t)
    
    small = t < 0.15
    mFactor[small] = 0.5*t[small] + 1
    
    large = t > 1.2
    tBig = t[large]
    mFactor[large] = tBig*np.expm1(1/tBig) * (1 - 3*np.exp(-2*tBig) + 5*np.exp(-6*tBig))
    
    justRight = ~small & ~large
    mFactor[justRight] = np.interp(t[justRight], rejectionT, rejectionM) + rejectionSafety * ((1.2-t[justRight])/1.1 + .1*(t[justRight]-0.1)/1.1)
    # I'm linearly decreasing rejectionSafety so that rejection(t) is closer to continuous at 1.2

    return mFactor
        
#the heat kernel distribution function, which selects from a number of approximations depending on cosTh and t
def hkDist(cosTh, t):
    """
    Practical S^2 heat-kernel density used for sampling.
    """
    cosTh = np.asarray(cosTh, dtype=float)
    cosTh, t = np.broadcast_arrays(cosTh, t)

    th = np.arccos(np.clip(cosTh, -1, 1))
    dist = np.empty_like(cosTh)

    # harmonic sum for large t
    largeT = t > tThresh
    dist[largeT] = hkSpectral(cosTh[largeT], t[largeT], lDist)

    # MPSD representation for small t
    smallT = ~largeT
    dist[smallT] = hkMPSD(th[smallT], t[smallT])

    # correct antipodal boundary layer where needed
    checkBL = smallT & (th > thCheck)

    if np.any(checkBL):
        useBL = np.zeros_like(checkBL, dtype=bool)
        useBL[checkBL] = th[checkBL] > thBL(t[checkBL])

        dist[useBL] = hkBL(th[useBL], t[useBL])

    return dist

def hkScore(cosTh, t):
    """
    The score of the heat kernel distribution function K(x | x_0), in the scalar form d(log K)/d(cosTh)
    """

    cosTh = np.asarray(cosTh, dtype=float)
    cosTh, t = np.broadcast_arrays(cosTh, t)

    th = np.arccos(np.clip(cosTh, -1, 1))
    score = np.empty_like(cosTh)

    # harmonic sum for large t
    largeT = t > tThreshScore
    score[largeT] = hkSpectralScore(cosTh[largeT], t[largeT], lScore)

    # MPSD representation for small t
    smallT = ~largeT
    score[smallT] = hkMPSDScore(th[smallT], t[smallT])

    # correct antipodal boundary layer where needed
    checkBL = smallT & (th > thCheckScore)

    if np.any(checkBL):
        useBL = np.zeros_like(checkBL, dtype=bool)
        useBL[checkBL] = th[checkBL] > thBLScore(t[checkBL])

        score[useBL] = hkBLScore(th[useBL], t[useBL])

    return score

#---------------------------------
# Spectral sum approximation (adaptive functions for scalar t are further below)

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

def hkSpectralScore(cosTh, t, lMax):
    """
    d/d(cos(theta)) log K for the truncated spectral heat kernel.
    """

    cosTh = np.asarray(cosTh, dtype=float)

    hk = np.ones_like(cosTh, dtype=float)
    dhk = np.zeros_like(cosTh, dtype=float)

    for ell in range(1, lMax + 1):

        coeff = (2*ell + 1)*np.exp(-ell*(ell + 1)*t)

        hk += coeff*eval_legendre(ell, cosTh)

        dhk += (
            coeff*(ell + 1)/2
            * eval_jacobi(ell - 1, 1, 1, cosTh)
        )

    return dhk/hk    

#---------------------------------
# MPSD approximation

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

    return hkMPSDLeading(th, t)*hkMPSDAsymp(th, t)
    
def hkMPSDLeading(th, t):
    """Leading order Minakshisundaram-Pleijel-Schwinger-DeWitt (MPSD) approximation to the heat kernel."""
    
    th = np.asarray(th)

    # np.sinc(th/pi) = sin(th)/th, and is well behaved at th = 0
    vanVleck = 1 / np.sqrt(np.sinc(th / np.pi))

    return (
        vanVleck
        * np.exp(-th**2 / (4 * t) + t / 4)
        / (4 * np.pi * t)
    )
    
def hkMPSDAsymp(th, t):
    """Asymptotic corrections in MPSD approximation up to O(t^4) counting th2 ~ O(t)."""
    
    th = np.asarray(th, dtype=float)
    th2 = th**2
    
    correction = (1
        + (1/12 + th2/180 + th2**2/1890 + th2**3/18900)*t
        + (7/480 + 13*th2/5040 + 19*th2**2/50400)*t**2
        + (31/8064 + 157*th2/120960)*t**3
        + 127*t**4/92160)

    return correction

def hkMPSDScore(th, t):
    """
    d/d(cos(theta)) log K for the leading order MPSD approximation.
    """

    th = np.asarray(th, dtype=float)

    tol = 1e-9
    small = np.abs(th) < tol

    # Replace tiny theta temporarily so the full expression never evaluates at zero.
    thSafe = np.where(small, tol, th)

    score = (
        thSafe/t
        - 1/thSafe
        + 1/np.tan(thSafe)
    )/(2*np.sin(thSafe))

    # Small-theta series through O(theta^2)
    scoreSmall = (
        1/(2*t) - 1/6
        + (1/(12*t) - 7/180)*th**2
    )
    
    return np.where(small, scoreSmall, score)

def hkMPSDScoreAsymp(th, t):
    """Asymptotic corrections to the score of the heat kernel up to O(t^3)."""

    th = np.asarray(th, dtype=float)
    th2 = th**2
    
    dCorrection = (
        (1/90 + 2*th2/945 + th2**2/3150)*t
        + (13/2520 + 19*th2/12600)*t**2
        + 157*t**3/60480
    )

    return -dCorrection/(hkMPSDAsymp(th,t)*np.sinc(th / np.pi))

#---------------------------------
# Boundary layer handling for short time

thBLT = np.array([0, 0.01, 0.02, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15], dtype=float)
thBLVals = np.array([np.pi, 3.07217, 3.03987, 3.02685, 2.97632, 2.93829, 2.90671, 2.87929, 2.85491], dtype=float)
thBLScoreT = np.array([0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.075, 0.1, 0.125, 0.137, 0.15], dtype=float)
thBLScoreVals = np.array([np.pi, 3.10136, 3.08069, 3.06392, 3.04923, 3.0359, 2.98034, 2.93508, 2.89601, 2.8615, 2.84627, 2.83064], dtype=float)
    
def thBL(t):
    """
    Boundary between the MPSD and antipodal boundary-layer regions.
    Calibrated for 0.01 <= t <= 0.15.
    """

    if np.any(t < 0):
        raise ValueError("t must be positive")

    elif np.any(t > thBLT[-1]):
        raise ValueError("thBL is only defined for the short-time regime t <= 0.15")

    else:
        return np.interp(t, thBLT, thBLVals)

def thBLScore(t):
    """
    Boundary between the MPSD and antipodal boundary-layer regions appropriate for score
    """

    if np.any(t < 0):
        raise ValueError("t must be positive")

    elif np.any(t > thBLScoreT[-1]):
        raise ValueError("thBL is only defined for the short-time regime t <= 0.15")

    else:
        return np.interp(t, thBLScoreT, thBLScoreVals)

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

    #elif method == "pi":
    #    return np.full_like(th, hkPi(t), dtype=float)

    elif method == "asymptotic":
        return hkBLasymptotic(th, t)

    else:
        raise ValueError(f"Unknown boundary-layer method: {method}")

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

def hkBLScore(th, t):
    """
    d/d(cos(theta)) log K for the boundary-layer approximation,
    including the first subleading correction.
    """

    th = np.asarray(th, dtype=float)

    delta = np.pi - th
    z = np.pi*delta/(2*t)

    i0 = i0e(z)
    i1 = i1e(z)

    # I1(z)/z -> 1/2 as z -> 0
    i1OverZ = np.full_like(z, 0.5, dtype=float)
    np.divide(i1, z, out=i1OverZ, where=np.abs(z) > 1e-12)

    denominator = (
        i0
        - t/np.pi**2 * (z**2*i0 + z*i1)
    )

    numeratorOverZ = (
        i1OverZ
        - t/np.pi**2 * (3*i0 + z*i1)
    )

    deltaOverSinDelta = 1/np.sinc(delta/np.pi)

    return (
        np.pi**2/(4*t**2)
        * deltaOverSinDelta
        * numeratorOverZ/denominator
    )

#---------------------------------
# Functions intended for use with scalar t
    
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
            hkDistScalar(cosTh, t)
            / (rejection(t)*vMFDist(cosTh, kappa))
        )

        if np.random.random() < acceptProb:
            return nProp


def hkDistScalar(cosTh, t):
    """
    Practical S^2 heat-kernel density used for sampling. cosTh is assumed to be a single scalar value
    """

    #cosTh = np.asarray(cosTh, dtype=float)
    th = np.arccos(np.clip(cosTh, -1, 1))

    # spectral representation for moderate and large diffusion times
    if t >= 0.15:
        return hkSpectralAdaptive(cosTh, t)

    # only check the boundary-layer cutoff in the tiny antipodal region
    thCheck = 2.90569
    if th > thCheck and th > thBL(t) :
        return hkBL(th, t)

    return hkMPSD(th, t)

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


# minimum t for which each lMax is sufficiently accurate
# (so lMax=0 is good enough for t > 6.2, lMax=1 is for 6.2 > t > 2.2 etc.)
# the minimum t for the score is set using an independent bound on the error
# So lMax=1 is good enough for t > 2.51, lMax=2 is for 2.51 > t > 1.06 etc.
spectralTMin = [6.2, 2.2, 1.1, 0.65, 0.44, 0.32, 0.24, 0.19, 0.15, 0.12, 0.1]
spectralTMinScore = [2.51, 1.06, 0.619, 0.423, 0.317, 0.252, 0.208, 0.177, 0.154, 0.135]

def lMaxValue(t):
    for lMax in range(len(spectralTMin)):
        if t >= spectralTMin[lMax]:
            return lMax

    raise ValueError(
        f"Spectral approximation not calibrated below t={spectralTMin[-1]}"
    )

def lMaxValueScore(t):
    for l in range(len(spectralTMinScore)):
        if t >= spectralTMin[l]:
            return l+1

    raise ValueError(
        f"Spectral approximation not calibrated below t={spectralTMin[-1]}"
    )
    
def hkSpectralAdaptive(cosTh, t):
    """
    Spectral S^2 heat kernel with lMax chosen from calibrated
    accuracy thresholds.
    """

    return hkSpectral(cosTh, t, lMaxValue(t))

def hkSpectralAdaptiveScore(cosTh, t):

    return hkSpectralScore(cosTh, t, lMaxValueScore(t))