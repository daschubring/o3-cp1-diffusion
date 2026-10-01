import torch
import math

#the boundary between large and short times for heat kernel distribution and score
tThresh = 0.15
tThreshScore = 0.137

#default harmonic truncations
lDist = 8
lScore = 10

#theta above which special handling takes place at short times
thCheck = 2.90569
thCheckScore = 2.84627

# ============================================================
# Basic vector geometry
# ============================================================

def norm(v):
    """Euclidean norm along the trailing vector axis, retaining that axis."""
    return torch.linalg.vector_norm(v, dim=-1, keepdim=True)


def normalize(v):
    """Normalize vectors along the trailing axis."""
    return v / norm(v)


def dot(v, w):
    """Dot product along the trailing vector axis, retaining that axis."""
    return torch.sum(v * w, dim=-1, keepdim=True)

def tangent(x, x0):
    """
    Tangent vector at x pointing towards x0. Magnitude sin theta
    """
    cosTh = dot(x, x0)
    return x0 - cosTh*x

def randomPerp(mUnit):
    """
    Sample a random unit vector perpendicular to each unit vector mUnit.

    mUnit has shape (..., 3). The returned tensor has the same shape,
    device, and dtype.
    """
    if mUnit.shape[-1] != 3:
        raise ValueError("mUnit must have shape (..., 3)")

    zVec = mUnit.new_tensor([0.0, 0.0, 1.0])
    yVec = mUnit.new_tensor([0.0, 1.0, 0.0])

    # Trial perpendicular direction.
    w = torch.cross(zVec.expand_as(mUnit), mUnit, dim=-1)

    # If mUnit is parallel to z, use y instead as the reference axis.
    wNorm = norm(w)
    parallelToZ = wNorm < 1e-6
    wAlt = torch.cross(yVec.expand_as(mUnit), mUnit, dim=-1)
    w = torch.where(parallelToZ, wAlt, w)

    # Orthonormal basis of the plane perpendicular to mUnit.
    wUnit = normalize(w)
    xUnit = torch.cross(mUnit, wUnit, dim=-1)

    # Uniform random angle in that plane.
    phi = 2.0 * torch.pi * torch.rand(
        mUnit.shape[:-1] + (1,), device=mUnit.device, dtype=mUnit.dtype
    )

    return torch.cos(phi) * wUnit + torch.sin(phi) * xUnit

# ============================================================
# Tedious torch stuff
# ============================================================
    
def broadcastFix(vector, scalar):
    """
    Broadcast a tensor of vectors (..., 3) and a scalar field (...).

    scalar may also have a trailing singleton dimension (..., 1).
    Returns shapes (..., 3) and (..., 1).
    """
    scalar = torch.as_tensor(
        scalar, device=vector.device, dtype=vector.dtype
    )

    if scalar.ndim > 0 and scalar.shape[-1] == 1:
        scalar = scalar.squeeze(-1)

    outShape = torch.broadcast_shapes(vector.shape[:-1], scalar.shape)

    vector = torch.broadcast_to(vector, outShape + (3,))
    scalar = torch.broadcast_to(scalar, outShape).unsqueeze(-1)

    return vector, scalar

def setDevice(device):
    """
    Move global tensors to GPU
    """
    global rejectionM
    global thBLT, thBLVals, thBLScoreT, thBLScoreVals

    rejectionM = rejectionM.to(device)

    thBLT = thBLT.to(device)
    thBLVals = thBLVals.to(device)
    thBLScoreT = thBLScoreT.to(device)
    thBLScoreVals = thBLScoreVals.to(device)

# ============================================================
# Sampling from von Mises-Fisher and the heat kernel
# ============================================================

# kSwitch is when to use the asymptotic form instead of an expression involving an exponential of \pm \kappa
kSwitch = 10.0

def vMF(mUnit, kappa):
    """
    Sample the von Mises-Fisher distribution on S^2.
    """
    if mUnit.shape[-1] != 3:
        raise ValueError("mUnit must have shape (..., 3)")

    mUnit, kappa = broadcastFix(mUnit, kappa)

    if torch.any(kappa < 0):
        raise ValueError("kappa must be nonnegative")

    outShape = mUnit.shape[:-1]

    chi = torch.rand(
        outShape + (1,),
        device=mUnit.device,
        dtype=mUnit.dtype
    )

    cosTh = torch.empty_like(kappa)

    zero = kappa == 0
    small = (kappa > 0) & (kappa <= kSwitch)
    large = kappa > kSwitch

    cosTh[zero] = 2.0 * chi[zero] - 1.0

    k = kappa[small]
    q = chi[small]
    cosTh[small] = -1.0 + torch.log1p(
        q * torch.expm1(2.0 * k)
    ) / k

    k = kappa[large]
    q = chi[large]
    cosTh[large] = 1.0 + torch.log(q) / k

    cosTh = torch.clamp(cosTh, -1.0, 1.0)
    sinTh = torch.sqrt(torch.clamp(1.0 - cosTh**2, min=0.0))

    return cosTh * mUnit + sinTh * randomPerp(mUnit)

def hk(mUnit, t):
    """
    Sample the S^2 heat kernel.
    """
    mUnit, t = broadcastFix(mUnit, t)

    if torch.any(t < 0):
        raise ValueError("t must be nonnegative")

    shape = mUnit.shape
    mUnit = mUnit.reshape(-1, 3)
    t = t.reshape(-1, 1)

    samples = torch.empty_like(mUnit)

    zero = (t.squeeze(-1) == 0)
    samples[zero] = mUnit[zero]

    positive = ~zero

    if positive.any():
        tPos = t[positive]
        mPos = mUnit[positive]

        kappa = 1.0 / (2.0 * tPos)

        # These indices are relative to the positive-time subset.
        remaining = torch.arange(
            len(mPos),
            device=mUnit.device
        )

        samplesPos = torch.empty_like(mPos)

        while remaining.numel() > 0:
            tLeft = tPos[remaining]
            kappaLeft = kappa[remaining]
            mLeft = mPos[remaining]

            nProp = vMF(mLeft, kappaLeft)
            cosTh = dot(nProp, mLeft)

            acceptProb = (
                hkDist(cosTh, tLeft)
                / (rejection(tLeft) * vMFDist(cosTh, kappaLeft))
            ).squeeze(-1)

            accepted = torch.rand_like(acceptProb) < acceptProb

            samplesPos[remaining[accepted]] = nProp[accepted]
            remaining = remaining[~accepted]

        samples[positive] = samplesPos

    return samples.reshape(shape)
    
# numerically optimized rejection-envelope values for 0.1 < t < 1.2
rejectionM = torch.tensor([
    1.042925, 1.064141, 1.081439, 1.092474, 1.097214, 1.096852,
    1.092859, 1.086554, 1.079008, 1.071053, 1.063342, 1.056405,
    1.050692, 1.046612, 1.044556, 1.044920, 1.048123, 1.054626,
    1.064952, 1.079711, 1.099626, 1.122410, 1.142114
], dtype=torch.float32)
rejectionSafety = 0.01

def rejection(t):

    mFactor = torch.empty_like(t)

    small = t < 0.15
    mFactor[small] = 1.0 + 0.5 * t[small]

    large = t > 1.2
    tBig = t[large]
    mFactor[large] = (
        tBig * torch.expm1(1.0 / tBig)
        * (1.0 - 3.0 * torch.exp(-2.0 * tBig)
                 + 5.0 * torch.exp(-6.0 * tBig))
    )

    justRight = ~small & ~large
    tMid = t[justRight]

    # Linear interpolation on rejectionM.
    # Table points are t = 0.1, 0.15, ..., 1.2.
    x = (tMid - 0.1) / 0.05
    i = torch.floor(x).long()
    i = torch.clamp(i, max=len(rejectionM) - 2)
    frac = x - i

    table = rejectionM

    interp = table[i] + frac * (table[i + 1] - table[i])

    safety = rejectionSafety * (
        (1.2 - tMid) / 1.1
        + 0.1 * (tMid - 0.1) / 1.1
    )

    mFactor[justRight] = interp + safety

    return mFactor

# ============================================================
# Distribution functions and scores
# ============================================================    

#normalized von Mises-Fisher density on S^2 as a function of cos(theta)
def vMFDist(cosTh, kappa):

    zero = kappa == 0
    dist = torch.empty_like(cosTh)

    dist[zero] = 1.0 / (4.0 * torch.pi)

    regular = ~zero
    k = kappa[regular]
    c = cosTh[regular]

    dist[regular] = (
        k * torch.exp(k * (c - 1.0))
        / (2.0 * torch.pi * (-torch.expm1(-2.0 * k)))
    )

    return dist

#the heat kernel distribution function, which selects from a number of approximations depending on cosTh and t
def hkDist(cosTh, t):
    """
    Practical S^2 heat-kernel density used for sampling.
    """
    cosTh, t = torch.broadcast_tensors(cosTh, t)

    th = torch.acos(torch.clamp(cosTh, -1.0, 1.0))
    dist = torch.empty_like(cosTh)

    largeT = t > tThresh
    dist[largeT] = hkSpectral(cosTh[largeT], t[largeT], lDist)

    smallT = ~largeT
    dist[smallT] = hkMPSD(th[smallT], t[smallT])

    checkBL = smallT & (th > thCheck)

    useBL = torch.zeros_like(checkBL)
    useBL[checkBL] = th[checkBL] > thBL(t[checkBL])

    dist[useBL] = hkBL(th[useBL], t[useBL])

    return dist

def hkScore(cosTh, t):
    """
    The score of the heat kernel distribution function K(x | x_0),
    in the scalar form d(log K)/d(cosTh).
    """

    cosTh, t = torch.broadcast_tensors(cosTh, t)

    th = torch.acos(torch.clamp(cosTh, -1.0, 1.0))
    score = torch.empty_like(cosTh)

    # harmonic sum for large t
    largeT = t > tThreshScore
    score[largeT] = hkSpectralScore(
        cosTh[largeT], t[largeT], lScore
    )

    # MPSD representation for small t
    smallT = ~largeT
    score[smallT] = hkMPSDScore(
        th[smallT], t[smallT]
    )

    # correct antipodal boundary layer where needed
    checkBL = smallT & (th > thCheckScore)

    useBL = torch.zeros_like(checkBL)
    useBL[checkBL] = th[checkBL] > thBLScore(t[checkBL])

    score[useBL] = hkBLScore(
        th[useBL], t[useBL]
    )

    return score

# ============================================================
# Spectral representation of the heat kernel
# ============================================================  
    
def hkSpectral(cosTh, t, lMax):
    """
    S^2 heat kernel from the spherical-harmonic expansion.
    """

    t = torch.as_tensor(t, device=cosTh.device, dtype=cosTh.dtype)
    cosTh, t = torch.broadcast_tensors(cosTh, t)

    hk = torch.ones_like(cosTh)

    if lMax == 0:
        return hk / (4.0 * torch.pi)

    pOld = torch.ones_like(cosTh)  # P_0
    p = cosTh                     # P_1

    q = torch.exp(-2.0 * t)

    # ell = 1
    decay = q
    qPower = q

    hk += 3.0 * p * decay

    for ell in range(1, lMax):

        pNew = (
            (2.0 * ell + 1.0) * cosTh * p
            - ell * pOld
        ) / (ell + 1.0)

        # qPower = q^(ell+1)
        qPower = qPower * q
        decay = decay * qPower

        hk += (2.0 * ell + 3.0) * pNew * decay

        pOld, p = p, pNew

    return hk / (4.0 * torch.pi)

def hkSpectralScore(cosTh, t, lMax):
    """
    d/d(cos(theta)) log K for the truncated spectral heat kernel.
    """

    t = torch.as_tensor(t, device=cosTh.device, dtype=cosTh.dtype)
    cosTh, t = torch.broadcast_tensors(cosTh, t)

    hk = torch.ones_like(cosTh)
    dhk = torch.zeros_like(cosTh)

    if lMax == 0:
        return dhk

    # P_0, P_1 and their derivatives
    pOld = torch.ones_like(cosTh)
    p = cosTh

    dpOld = torch.zeros_like(cosTh)
    dp = torch.ones_like(cosTh)

    q = torch.exp(-2.0 * t)

    decay = q
    qPower = q

    # ell = 1
    coeff = 3.0 * decay
    hk += coeff * p
    dhk += coeff * dp

    for ell in range(1, lMax):
        pNew = (
            (2.0 * ell + 1.0) * cosTh * p
            - ell * pOld
        ) / (ell + 1.0)

        dpNew = (
            (2.0 * ell + 1.0) * (p + cosTh * dp)
            - ell * dpOld
        ) / (ell + 1.0)

        qPower = qPower * q
        decay = decay * qPower

        coeff = (2.0 * ell + 3.0) * decay

        hk += coeff * pNew
        dhk += coeff * dpNew

        pOld, p = p, pNew
        dpOld, dp = dp, dpNew

    return dhk / hk    

# ============================================================
# MPSD approximation to the heat kernel
# ============================================================  

def hkMPSD(th, t):
    """
    Short-time MPSD approximation on S^2, including corrections
    through O(t^4) with theta powers counted as theta^2 ~ t.
    """

    # It is sufficient up to the ~10^{-6} integrated error we use throughout to use
    # O(t^4) for 0.10 < t < 0.15
    # O(t^3) for 0.05 < t < 0.10
    # O(t^2) for 0.02 < t < 0.05
    # O(t) for t < 0.02
    # But since O(t^4) is inexpensive I use it for all t < 0.15.

    return hkMPSDLeading(th, t) * hkMPSDAsymp(th, t)


def hkMPSDLeading(th, t):
    """
    Leading order Minakshisundaram-Pleijel-Schwinger-DeWitt
    approximation to the heat kernel.
    """

    # torch.sinc(th/pi) = sin(th)/th, and is well behaved at th = 0
    vanVleck = 1.0 / torch.sqrt(torch.sinc(th / torch.pi))

    return (
        vanVleck
        * torch.exp(-th**2 / (4.0 * t) + t / 4.0)
        / (4.0 * torch.pi * t)
    )


def hkMPSDAsymp(th, t):
    """
    Asymptotic corrections in MPSD approximation up to O(t^4),
    counting theta^2 ~ O(t).
    """

    th2 = th**2

    correction = (
        1.0
        + (1/12 + th2/180 + th2**2/1890 + th2**3/18900) * t
        + (7/480 + 13*th2/5040 + 19*th2**2/50400) * t**2
        + (31/8064 + 157*th2/120960) * t**3
        + 127*t**4/92160
    )

    return correction


def hkMPSDScore(th, t):
    """
    d/d(cos(theta)) log K for the MPSD approximation.
    """

    tol = 5e-2
    small = torch.abs(th) < tol

    thSafe = torch.where(small, torch.ones_like(th), th)

    scoreLeading = (
        thSafe/t
        - 1.0/thSafe
        + 1.0/torch.tan(thSafe)
    ) / (2.0*torch.sin(thSafe))

    scoreLeadingSmall = (
        1.0/(2.0*t) - 1.0/6.0
        + (1.0/(12.0*t) - 7.0/180.0)*th**2
    )

    scoreLeading = torch.where(
        small, scoreLeadingSmall, scoreLeading
    )

    return scoreLeading + hkMPSDScoreAsymp(th, t)


def hkMPSDScoreAsymp(th, t):
    """
    Asymptotic corrections to the score of the heat kernel up to O(t^3).
    """

    th2 = th**2

    dCorrection = (
        (1/90 + 2*th2/945 + th2**2/3150) * t
        + (13/2520 + 19*th2/12600) * t**2
        + 157*t**3/60480
    )

    return -dCorrection / (
        hkMPSDAsymp(th, t) * torch.sinc(th / torch.pi)
    )

# ============================================================
# Boundary layer handling for short time
# ============================================================  

thBLT = torch.tensor(
    [0, 0.01, 0.02, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
    dtype=torch.float32
)

thBLVals = torch.tensor(
    [torch.pi, 3.07217, 3.03987, 3.02685, 2.97632,
     2.93829, 2.90671, 2.87929, 2.85491],
    dtype=torch.float32
)

thBLScoreT = torch.tensor(
    [0, 0.005, 0.01, 0.015, 0.02, 0.025,
     0.05, 0.075, 0.1, 0.125, 0.137, 0.15],
    dtype=torch.float32
)

thBLScoreVals = torch.tensor(
    [torch.pi, 3.10136, 3.08069, 3.06392, 3.04923, 3.0359,
     2.98034, 2.93508, 2.89601, 2.8615, 2.84627, 2.83064],
    dtype=torch.float32
)


def _interp1d(t, x, y):
    """Linear interpolation on a tabulated 1D grid."""
    i = torch.bucketize(t, x, right=True) - 1
    i = torch.clamp(i, 0, len(x) - 2)

    frac = (t - x[i]) / (x[i + 1] - x[i])

    return y[i] + frac * (y[i + 1] - y[i])


def thBL(t):
    """
    Boundary between the MPSD and antipodal boundary-layer regions.
    Defined for the short-time regime 0 <= t <= 0.15.
    """
    return _interp1d(t, thBLT, thBLVals)


def thBLScore(t):
    """
    Boundary between the MPSD and antipodal boundary-layer regions
    appropriate for the score.
    """
    return _interp1d(t, thBLScoreT, thBLScoreVals)


def hkBL(th, t, method="zero"):
    """
    Handle the antipodal boundary-layer region of the S^2 heat kernel.

    method:
        "zero"       : discard the exponentially suppressed region
        "asymptotic" : use the boundary-layer asymptotic expansion
    """

    if method == "zero":
        return torch.zeros_like(th)

    elif method == "asymptotic":
        return hkBLAsymptotic(th, t)

    else:
        raise ValueError(f"Unknown boundary-layer method: {method}")


def hkBLAsymptotic(th, t):
    """
    Boundary-layer asymptotic approximation near theta = pi,
    including the first subleading correction.
    """

    th, t = torch.broadcast_tensors(th, t)

    delta = torch.pi - th
    z = torch.pi * delta / (2.0 * t)

    i0 = torch.special.i0e(z)
    i1 = torch.special.i1e(z)

    prefactor = (
        math.sqrt(math.pi)
        / (4.0 * t**1.5)
        * torch.exp(
            t/4.0
            - torch.pi**2/(4.0*t)
            + torch.abs(z)
        )
    )

    leading = i0

    correction = (
        t/torch.pi**2
        * (z**2*i0 + z*i1)
    )

    return prefactor * (leading - correction)


def hkBLScore(th, t):
    """
    d/d(cos(theta)) log K for the boundary-layer approximation,
    including the first subleading correction.
    """

    th, t = torch.broadcast_tensors(th, t)

    delta = torch.pi - th
    z = torch.pi * delta / (2.0 * t)

    i0 = torch.special.i0e(z)
    i1 = torch.special.i1e(z)

    # i1e(z)/z -> 1/2 as z -> 0.
    zero = z == 0
    zSafe = torch.where(zero, torch.ones_like(z), z)
    i1OverZ = i1 / zSafe
    i1OverZ = torch.where(
        zero,
        torch.full_like(z, 0.5),
        i1OverZ
    )

    denominator = (
        i0
        - t/torch.pi**2 * (z**2*i0 + z*i1)
    )

    numeratorOverZ = (
        i1OverZ
        - t/torch.pi**2 * (3.0*i0 + z*i1)
    )

    deltaOverSinDelta = 1.0 / torch.sinc(delta / torch.pi)

    return (
        torch.pi**2/(4.0*t**2)
        * deltaOverSinDelta
        * numeratorOverZ/denominator
    )

# ============================================================
# Adaptive spectral representation of the heat kernel, for use with scalar t
# ============================================================  

# minimum t for which each lMax is sufficiently accurate
# (so lMax=0 is good enough for t > 6.2, lMax=1 is for 6.2 > t > 2.2 etc.)
# the minimum t for the score is set using an independent bound on the error
# So lMax=1 is good enough for t > 2.51, lMax=2 is for 2.51 > t > 1.06 etc.
spectralTMin = [6.2, 2.2, 1.1, 0.65, 0.44, 0.32, 0.24, 0.19, 0.15, 0.12, 0.1]
spectralTMinScore = [2.51, 1.06, 0.619, 0.423, 0.317, 0.252, 0.208, 0.177, 0.154, 0.135]

def lMaxValue(t):
    """
    Minimum lMax calibrated for a given scalar diffusion time t.
    """
    for lMax, tMin in enumerate(spectralTMin):
        if t >= tMin:
            return lMax

    raise ValueError(
        f"Spectral approximation not calibrated below t={spectralTMin[-1]}"
    )

def hkSpectralAdaptive(cosTh, t):
    """
    Adaptive version intended for scalar/shared t.
    """
    return hkSpectral(cosTh, t, lMaxValue(float(t)))

def lMaxValueScore(t):
    for l in range(len(spectralTMinScore)):
        if t >= spectralTMinScore[l]:
            return l + 1

    raise ValueError(
        f"Spectral approximation not calibrated below t={spectralTMinScore[-1]}"
    )

def hkSpectralAdaptiveScore(cosTh, t):
    return hkSpectralScore(cosTh, t, lMaxValueScore(t))