import torch
import sphere

t0 = 0.005
tF = 4.0

def tauFromU(u, tauI=t0, tauF=tF):
    """Return the diffusion time tau from our normalized sampling time u."""
    u = torch.as_tensor(u)
    return tauI * (torch.pow(1 + tauF / tauI, u) - 1)

def sigmaFromU(u, tauI=t0, tauF=tF):
    """Standard deviation from u."""
    u = torch.as_tensor(u)
    return torch.sqrt(2 * tauFromU(u, tauI, tauF))

def uFromTau(tau, tauI=t0, tauF=tF):
    """Return our normalized sampling time u from the diffusion time tau."""
    tau = torch.as_tensor(tau)
    return torch.log(tau / tauI + 1) / torch.log(
        torch.as_tensor(tauF / tauI + 1, device=tau.device, dtype=tau.dtype)
    )

def expMap(x, v):
    """Take the exponential map of a vector v in the tangent space of the sphere at point x.

    x is a 3 component unit vector and v is orthogonal to it.
    """
    r = sphere.norm(v)
    xNew = torch.cos(r) * x + torch.sinc(r / torch.pi) * v
    return xNew

def eulerStep(x, t, dt, score):
    """A step of an Euler ODE integrator."""
    v = dt * score(x, t)
    return expMap(x, v)

def rk2Step(x, t, dt, score):
    """
    The STVDRK2 method of Leung, Chau, Lee (2024).
    A natural S^2 version of Heun.
    """

    # Do two forward Euler steps
    x1 = eulerStep(x, t, dt, score)
    x2 = eulerStep(x1, t - dt, dt, score)

    # Interpolate back one step
    return sphere.normalize(x + x2)

def eulerMaruyamaStep(x, t, dt, score):
    """
    A step of the Euler-Maruyama method for the reverse diffusion SDE.
    """
    xi = torch.randn_like(x)
    xi = xi - sphere.dot(xi, x) * x

    v = 2 * dt * score(x, t) + torch.sqrt(torch.as_tensor(2 * dt, device=x.device, dtype=x.dtype)) * xi
    xNew = expMap(x, v)

    return xNew

def langevinStep(x, t, ds, score):
    """
    A step of a Langevin SDE at fixed diffusion time t.
    """
    xi = torch.randn_like(x)
    xi = xi - sphere.dot(xi, x) * x

    v = ds * score(x, t) + torch.sqrt(torch.as_tensor(2 * ds, device=x.device, dtype=x.dtype)) * xi
    xNew = expMap(x, v)

    return xNew

def quenchStep(x, t, ds, score):
    """
    Drift of x in the direction of score. Langevin with no noise.
    """

    v = ds * score(x, t)
    xNew = expMap(x, v)

    return xNew
