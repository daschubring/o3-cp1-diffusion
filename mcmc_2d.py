import torch

import sphere


def randomSpins(nCfg, L, device=None, dtype=torch.float32):
    """Generate nCfg independent random spin configurations on S^2."""
    spins = torch.randn((nCfg, L, L, 3), device=device, dtype=dtype)
    return sphere.normalize(spins)
    
def constantSpins(nCfg, L, device=None, dtype=torch.float32):
    """Generate nCfg constant spin configurations on S^2."""
    spins = torch.randn((nCfg, 3), device=device, dtype=dtype)
    spins = sphere.normalize(spins).reshape((nCfg, 1, 1, 3))
    return spins.expand(nCfg, L, L, 3).clone()


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

def heatBathSweep(spins, beta, fixed=False):

    A = spins[..., 0::2, 0::2, :]
    B = spins[..., 0::2, 1::2, :]
    C = spins[..., 1::2, 0::2, :]
    D = spins[..., 1::2, 1::2, :]

    # ----- even checkerboard: A and D -----

    mA = (
        B
        + torch.roll(B, 1, dims=-2)
        + C
        + torch.roll(C, 1, dims=-3)
    )

    mD = (
        C
        + torch.roll(C, -1, dims=-2)
        + B
        + torch.roll(B, -1, dims=-3)
    )

    newA = heatBathStep(mA, beta)

    if fixed:
        # Fine-lattice sites (4m,4n) are A[..., ::2, ::2, :]
        A[..., 0::2, 1::2, :] = newA[..., 0::2, 1::2, :]
        A[..., 1::2, 0::2, :] = newA[..., 1::2, 0::2, :]
        A[..., 1::2, 1::2, :] = newA[..., 1::2, 1::2, :]
    else:
        A[...] = newA

    D[...] = heatBathStep(mD, beta)

    # ----- odd checkerboard: B and C -----

    mB = (
        A
        + torch.roll(A, -1, dims=-2)
        + D
        + torch.roll(D, 1, dims=-3)
    )

    mC = (
        A
        + torch.roll(A, -1, dims=-3)
        + D
        + torch.roll(D, 1, dims=-2)
    )

    B[...] = heatBathStep(mB, beta)
    C[...] = heatBathStep(mC, beta)

    return spins

from tqdm.notebook import tqdm

def heatBath(spins, beta, nSweeps=1, fixed=False):
    """Perform nSweeps checkerboard heat-bath sweeps in place."""
    for _ in tqdm(range(nSweeps)):
        heatBathSweep(spins, beta, fixed=fixed)
    return spins

def correlation(cfgs):
    """
    Axis-averaged spin correlation C(r) for a periodic 2D square lattice.

    cfgs has shape (..., L, L, 3).

    Averages over:
      - all configurations / leading batch dimensions
      - all lattice sites
      - both spatial directions
    """
    # OLD VERSION (new FFT version is much quicker):
    # L = cfgs.shape[-2]

    # corr = torch.empty(L, device=cfgs.device, dtype=cfgs.dtype)

    # for r in range(L):
    #     corrX = torch.sum(
    #         torch.roll(cfgs, -r, dims=-2) * cfgs,
    #         dim=-1
    #     ).mean()

    #     corrY = torch.sum(
    #         torch.roll(cfgs, -r, dims=-3) * cfgs,
    #         dim=-1
    #     ).mean()

    #     corr[r] = 0.5 * (corrX + corrY)

    # return corr

    corr = corrMatrix(cfgs)
    return .5*(corr[0]+corr.T[0])

def corrMatrix(x):
    """
        Full L x L matrix of correlations
    """
    B, L, _, _ = x.shape

    # Fourier transform over the two lattice directions
    f = torch.fft.fft2(x, dim=(-3, -2))

    # sum power over O(3) components
    power = (f.abs()**2).sum(dim=-1)

    # inverse transform gives correlation at every displacement
    C = torch.fft.ifft2(power, dim=(-2, -1)).real

    return C.mean(dim=0) / (L * L)

def exactScore(x, beta):
    """The exact score at a field configuration x, assuming the equilibrium distribution """
    
    #Construct nearest neighbor field
    m = torch.roll(x, 1, dims=-2)+torch.roll(x, -1, dims=-2)+torch.roll(x, 1, dims=-3)+torch.roll(x, -1, dims=-3)
    
    #Return tangent component times beta
    return beta*(m - sphere.dot(m, x)*x)

def halfArea(a, b, c):
    tripProd = sphere.dot(a, torch.linalg.cross(b, c, dim=-1))

    denom = (
        1
        + sphere.dot(a, b)
        + sphere.dot(b, c)
        + sphere.dot(c, a)
    )

    return torch.atan2(tripProd, denom)

def topCharge(cfgs):
    a = cfgs
    b = torch.roll(cfgs, -1, dims=-3)
    c = torch.roll(cfgs, -1, dims=-2)
    d = torch.roll(b,    -1, dims=-2)
    
    area1 = halfArea(a, b, c)
    area2 = halfArea(d, c, b)
    
    area3 = halfArea(a, d, c)
    area4 = halfArea(a, b, d)
    
    Q1 = torch.sum(area1 + area2, dim=(-3, -2)).squeeze(-1) / (2*torch.pi)
    Q2 = torch.sum(area3 + area4, dim=(-3, -2)).squeeze(-1) / (2*torch.pi)
    defects = torch.sum(torch.abs(area1 + area2 - area3 - area4), dim=(-3, -2)).squeeze(-1) / (2*torch.pi)

    return Q1, Q2, defects

def magnetization(x):
    """
    Magnetization vector per site.

    x: (..., L, L, 3)
    returns: (..., 3)
    """
    return x.mean(dim=(-3, -2))


def susceptibility(x):
    """
    Reduced magnetic susceptibility:
        chi = <|M|^2>/V = V <|m|^2>

    x: (N, L, L, 3)
    returns: scalar
    """
    Lx, Ly = x.shape[-3:-1]
    V = Lx * Ly

    m = magnetization(x)
    return V * (m**2).sum(dim=-1).mean()

def correlationLength(x):
    """
    Finite-volume second-moment correlation length.

    x: (N, L, L, 3)
    """
    N, Lx, Ly, _ = x.shape
    assert Lx == Ly
    L = Lx
    V = L * L

    # Fourier transform over spatial coordinates
    sx = torch.fft.fftn(x, dim=(1, 2))

    # G(p) = <|S(p)|^2> / V
    G = (sx.abs()**2).sum(dim=-1).mean(dim=0) / V

    G0 = G[0, 0]

    # Average the two minimum nonzero lattice momenta
    Gmin = 0.5 * (G[1, 0] + G[0, 1])

    xi = torch.sqrt(G0 / Gmin - 1.0) / (2.0 * torch.sin(
        torch.tensor(torch.pi / L, device=x.device)
    ))

    return xi

def gaussianInterpSetup(corr):
    """
    Precompute everything needed for fast Gaussian interpolation.

    corr: (64,64) dot-product covariance C(dx,dy)

    Returns:
        sqrtSpectrum : Fourier-space sqrt covariance for one Cartesian component
        M            : conditioning matrix K_{.C} K_CC^{-1}
    """
    L = corr.shape[0]
    device = corr.device
    dtype = corr.dtype

    # ------------------------------------------------------------
    # 1. Fourier spectrum of one Cartesian component.
    #
    # <s_i^a s_j^b> = (1/3) K_ij delta_ab
    # ------------------------------------------------------------

    spectrum = torch.fft.fft2(corr).real / 3.0

    # Protect against tiny negative eigenvalues from roundoff/statistics
    spectrum = torch.clamp(spectrum, min=0.0)

    sqrtSpectrum = torch.sqrt(spectrum)


    # ------------------------------------------------------------
    # 2. Build K_{.C} and K_CC
    # ------------------------------------------------------------

    i, j = torch.meshgrid(
        torch.arange(L, device=device),
        torch.arange(L, device=device),
        indexing='ij'
    )

    coords = torch.stack((i.flatten(), j.flatten()), dim=1)

    coarse = (
        (coords[:, 0] % 4 == 0)
        & (coords[:, 1] % 4 == 0)
    )

    coarse_idx = torch.where(coarse)[0]

    # displacement from every fine site to every coarse site
    d = (coords[:, None, :] - coords[coarse_idx][None, :, :]) % L

    # K_{.C}: (4096,256)
    KallC = corr[d[..., 0], d[..., 1]]

    # K_CC: select coarse rows
    KCC = KallC[coarse_idx]

    # M = K_{.C} K_CC^{-1}
    #
    # Solve K_CC M^T = K_{.C}^T rather than explicitly invert.
    M = torch.linalg.solve(KCC, KallC.T).T

    return sqrtSpectrum, M

def gaussianInterpolate(coarse, sqrtSpectrum, M):
    """
    Fast conditional-Gaussian interpolation.

    coarse: (B,16,16,3)

    returns:
        x: (B,64,64,3)
    """

    B = coarse.shape[0]
    L = 64
    device = coarse.device
    dtype = coarse.dtype

    # ------------------------------------------------------------
    # 1. Draw unconditional Gaussian field z ~ N(0,K/3)
    #
    # Start with real white noise. FFT -> multiply by sqrt spectrum
    # -> inverse FFT.
    # ------------------------------------------------------------

    noise = torch.randn(B, L, L, 3, device=device, dtype=dtype)

    noiseK = torch.fft.fft2(noise, dim=(-3, -2))

    z = torch.fft.ifft2(
        noiseK * sqrtSpectrum[None, :, :, None],
        dim=(-3, -2)
    ).real


    # ------------------------------------------------------------
    # 2. Measure mismatch at coarse sites
    # ------------------------------------------------------------

    zCoarse = z[:, ::4, ::4, :]

    delta = coarse - zCoarse

    # (B,256,3)
    delta = delta.reshape(B, -1, 3)


    # ------------------------------------------------------------
    # 3. Propagate the correction to every site
    #
    # correction = K_{.C} K_CC^{-1} (x_C - z_C)
    # ------------------------------------------------------------

    correction = torch.einsum(
        'ij,bjk->bik',
        M,
        delta
    )

    # ------------------------------------------------------------
    # 4. Add correction
    # ------------------------------------------------------------

    x = z.reshape(B, L * L, 3) + correction
    x = x.reshape(B, L, L, 3)

    # Mathematically redundant, but guarantees exact anchor values
    # rather than ~1e-6 numerical agreement.
    x[:, ::4, ::4, :] = coarse

    return x