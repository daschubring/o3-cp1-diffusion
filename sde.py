import numpy as np
import sphere

def tauFromU(u, tau0, tauF):
    """ Return the diffusion time tau from our normalized sampling time u """
    return tau0*(np.power(1+tauF/tau0, u)-1)

def sigmaFromU(u, tau0, tauF):
    """ standard deviation from u """
    return np.sqrt(2*tauFromU(u, tau0, tauF))

def uFromTau(tau, tau0, tauF):
    """ Return our normalized sampling time u from the diffusion time tau """
    return np.log(tau/tau0 + 1)/np.log(tauF/tau0 + 1)

def expMap(x, v):
    """ Take the exponential map of a vector v in the tangent space of the sphere at point x
        x is a 3 component unit vector and v is orthogonal to it
    """
    r = sphere.norm(v)
    xNew = np.cos(r) * x + np.sinc(r / np.pi) * v
    return xNew

def eulerStep(x, t, dt, score):
    """ A step of an Euler ODE integrator """
    v = dt* score(x, t)
    return expMap(x, v)

def rk2Step(x, t, dt, score):
    """
    The STVDRK2 method of Leung, Chau, Lee (2024).
    A natural S^2 version of Heun
    """

    #Do two forward Euler steps
    x1 = eulerStep(x, t, dt, score)
    x2 = eulerStep(x1, t - dt, dt, score)

    #Interpolate back one step
    return sphere.normalize(x + x2)

def eulerMaruyamaStep(x, t, dt, score):
    """
    A step of the Euler-Maruyama method for the reverse diffusion SDE
    """
    xi = np.random.normal(size=x.shape)
    xi = xi - sphere.dot(xi, x) * x

    v = 2*dt*score(x, t) + np.sqrt(2*dt)*xi
    xNew = expMap(x, v)

    return xNew

def langevinStep(x, t, ds, score):
    """
    A step of a Langevin SDE at fixed diffusion time t
    """
    xi = np.random.normal(size=x.shape)
    xi = xi - sphere.dot(xi, x) * x

    v = ds*score(x, t) + np.sqrt(2*ds)*xi
    xNew = expMap(x, v)

    return xNew
    