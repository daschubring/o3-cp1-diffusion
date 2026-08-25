import numpy as np
import matplotlib.pyplot as plt
import sphere
import sde
import torch
import importlib
from model import MLP
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#CONSTRUCT INITIAL DISTRIBUTION

t0 = .005
tF = 4.
phEyes = 3*np.pi/8
phSmile = 3*np.pi/4
thEyeL = 3*np.pi/8
thEyeR = 5*np.pi/8
thSmileL = np.pi/4
thSmileR = 3*np.pi/4
nSmile = 13

#Eyes at specified positions
mEyeL = np.array([-np.sin(thEyeL)*np.sin(phEyes), np.cos(thEyeL), np.sin(thEyeL)*np.cos(phEyes)])
mEyeR = np.array([-np.sin(thEyeR)*np.sin(phEyes), np.cos(thEyeR), np.sin(thEyeR)*np.cos(phEyes)])

#Smile from nSmile points running from the L to R edges
thSmile = np.linspace(thSmileL, thSmileR, nSmile)

mSmile = np.empty((nSmile, 3), dtype=float)
for i in range(nSmile):
    mSmile[i] =  np.array([-np.sin(thSmile[i])*np.sin(phSmile), np.cos(thSmile[i]), np.sin(thSmile[i])*np.cos(phSmile)])

#SAMPLING FUNCTIONS

def generateInitial(m, t=t0):
    """Deprecated function, use smileySample instead.
    Samples from initial distribution, with fixed probability mass in the eyes and smile."""
   
    #Create initial batch
    #There will be nBatch = 8*m*nSmile unit vectors

    eyeLBatch = np.full((m*nSmile, 3), mEyeL)
    eyeRBatch = np.full((m*nSmile, 3), mEyeR)

    batches = [eyeLBatch, eyeRBatch]

    for i in range(nSmile):
        batches.append(np.full((6*m, 3), mSmile[i]))

    mUnit = np.concatenate(batches, axis=0)

    return sphere.hk(mUnit, t0)

def smileySample(batchSize):
    """Produces batchSize samples from the initial distribution"""
    
    #the probabilities of drawing from the 2 eyes at index 0 and 1, or the nSmile smile points
    prob = np.full(nSmile+2, 3./(4.*nSmile))
    prob[0] = 1./8
    prob[1] = 1./8

    #the number of samples drawn from the nSmile+2 individual eye and smile points
    counts = np.random.multinomial(batchSize, prob)

    #create initial mUnit for smiley
    eyeLBatch = np.full((counts[0], 3), mEyeL)
    eyeRBatch = np.full((counts[1], 3), mEyeR)

    batches = [eyeLBatch, eyeRBatch]

    for i in range(nSmile):
        batches.append(np.full((counts[i + 2], 3), mSmile[i]))

    mUnit = np.concatenate(batches, axis=0)

    #sample from initial smiley
    return sphere.hk(mUnit, t0)


def smileyDist(x, t):
    """Evaluate smiley distribution at point x at diffusion time t"""

    # Construct distribution
    cosThEyeL = sphere.dot(x, mEyeL)
    cosThEyeR = sphere.dot(x, mEyeR)
    
    #Add an extra index to x in the -2 position to handle the individual components of smile
    cosThSmile = sphere.dot(x[..., None, :], mSmile)

    #Evaluate individual heat kernels
    hkEyeL = sphere.hkDist(cosThEyeL, t0+t)
    hkEyeR = sphere.hkDist(cosThEyeR, t0+t)
    hkSmile = sphere.hkDist(cosThSmile, t0+t)

    #Return weighted sum
    return (.75/nSmile)*hkSmile.sum(-2) + (.25/2)*(hkEyeL+hkEyeR)

def scoreSmiley(x, t):
    """Evaluate score of smiley distribution at point x at diffusion time t"""
    
    t = np.atleast_1d(t)
    
    #Add an extra index to x in the -2 position to handle the individual components of smile
    xSmile = x[..., None, :]
    tSmile = t[..., None, :]
    
    cosThEyeL = sphere.dot(x, mEyeL)
    cosThEyeR = sphere.dot(x, mEyeR)
    cosThSmile = sphere.dot(xSmile, mSmile)

    #calculate individual scores from source points in distribution
    scoreEyeL = sphere.hkScore(cosThEyeL, t0+t)*sphere.tangent(x, mEyeL)
    scoreEyeR = sphere.hkScore(cosThEyeR, t0+t)*sphere.tangent(x, mEyeR)
    scoreSmile = sphere.hkScore(cosThSmile, t0+tSmile)*sphere.tangent(xSmile, mSmile)

    #Evaluate individual weighted heat kernels
    wEyeL = .125*sphere.hkDist(cosThEyeL, t0+t)
    wEyeR = .125*sphere.hkDist(cosThEyeR, t0+t)
    wSmile = .75*sphere.hkDist(cosThSmile, t0+tSmile)

    smileyNorm = wSmile.mean(-2) + wEyeL + wEyeR

    #Average the scores
    return (wEyeL*scoreEyeL + wEyeR*scoreEyeR + np.mean(wSmile*scoreSmile, axis=-2))/smileyNorm


# PLOTTING FUNCTIONS

def sphereHistogram(samples, nPhi=100, nZ=100):
    phi = np.arctan2(samples[:, 1], samples[:, 0])
    phi %= 2*np.pi
    z = samples[:, 2]

    phiEdges = np.linspace(0, 2*np.pi, nPhi + 1)
    zEdges = np.linspace(-1, 1, nZ + 1)

    hist, _, _ = np.histogram2d(
        phi, z,
        bins=(phiEdges, zEdges),
        density=True
    )

    return hist.T, phiEdges, zEdges

def plotSphereHist(samples, nPhi=100, nZ=100, title=None):
    H, phiEdges, zEdges = sphereHistogram(samples, nPhi, nZ)

    plt.figure()
    plt.pcolormesh(phiEdges, zEdges, H, shading='auto')
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$z$')
    plt.colorbar()

    if title is not None:
        plt.title(title)

    plt.show()

# Plot distribution
def plotSphereDist(funct, nPhi=100, nZ=100, title=None):

    phi = np.linspace(0, 2*np.pi, nPhi, endpoint=False)
    z = np.linspace(-1, 1, nZ)[:, None]
    sinTh = np.sqrt(1 - z**2)

    vX = np.cos(phi)*sinTh
    vY = np.sin(phi)*sinTh
    vZ = np.broadcast_to(z, (nZ, nPhi))
    v = np.stack([vX, vY, vZ], axis=-1) 

    values = funct(v)

    plt.figure()
    plt.imshow(values, origin='lower', extent=[0, 2*np.pi, -1, 1])
    plt.colorbar(orientation='horizontal');

#Calculate TV distance of samples from exact distribution at t

def distHist(samples, nPhi, t=0):
    nZ = nPhi+1

    phi = np.linspace(0, 2*np.pi, nPhi, endpoint=False)
    z = np.linspace(-1, 1, nZ)[:, None]
    sinTh = np.sqrt(1 - z**2)

    vX = np.cos(phi)*sinTh
    vY = np.sin(phi)*sinTh
    vZ = np.broadcast_to(z, (nZ, nPhi))
    v = np.stack([vX, vY, vZ], axis=-1) 

    values = smileyDist(v, t)

    sampleHist = sphereHistogram(samples, nPhi, nZ)[0]

    integrand = .5*np.abs(values[...,0]-sampleHist)
    integrandZ = 2*np.pi*integrand.mean(1)

    #Trapezoidal rule
    dz = 2/(nZ-1)
    integral = dz*(.5*(integrandZ[0]+integrandZ[-1])+integrandZ[1:-1].sum())

    return integral


# CREATE TRAINING DATA

def sampleTrainingUniform(batchSize, multTimes = True, uTest = .5, minU = 0., maxU = 1., device = 'cpu'):
    """
    Create training data sampled uniformly over S^2, using exact score
    """
    
    #create random diffusion times tau
    if multTimes:
        u_np = (maxU - minU)*np.random.random((batchSize, 1))+minU
    else:
        u_np = np.full((batchSize, 1), uTest)

    tau = sde.tauFromU(u_np, t0, tF)

    #create uniform samples
    x_np = sphere.normalize(np.random.normal(0,1,(batchSize, 3)))

    #calculate exact score
    s_np = scoreSmiley(x_np, tau)
    
    #create and return torch tensors
    u = torch.from_numpy(uTest).float().to(device)
    x = torch.from_numpy(x_np).float().to(device)
    s = torch.from_numpy(s_np).float().to(device)
    
    return u, x, s


def sampleTrainingDirect(batchSize, multTimes = True, uTest = .5, minU = 0., maxU = 1., device = 'cpu'):
    """
    Create training data sampled according to the diffused smiley distribution, using exact score
    """

    #create random diffusion times tau
    if multTimes:
        u_np = (maxU - minU)*np.random.random((batchSize, 1))+minU
    else:
        u_np = np.full((batchSize, 1), uTest)
        
    tau = sde.tauFromU(u_np, t0, tF)

    #sample from smiley at time tau
    x0_np = smileySample(batchSize)
    x_np = sphere.hk(x0_np, tau)

    #calculate score
    s_np = scoreSmiley(x_np, tau)
    
    #create and return torch tensors
    u = torch.from_numpy(u_np).float().to(device)
    x = torch.from_numpy(x_np).float().to(device)
    s = torch.from_numpy(s_np).float().to(device)
    
    return u, x, s

def sampleTraining(batchSize, multTimes = True, uTest = .5, minU = 0., maxU = 1., device = 'cpu'):
    """
    Create training data sampled according to the diffused smiley distribution,
    The conditional score is used which doesn't require exact knowledge of the distribution
    """

    #create random diffusion times tau
    if multTimes:
        u_np = (maxU - minU)*np.random.random((batchSize, 1))+minU
    else:
        u_np = np.full((batchSize, 1), uTest)
        
    tau = sde.tauFromU(u_np, t0, tF)

    #sample from smiley
    x0_np = smileySample(batchSize)

    #create noised samples
    x_np = sphere.hk(x0_np, tau)

    #find score directions
    cosTh = sphere.dot(x0_np, x_np)
    tangent = x0_np - cosTh*x_np

    #calculate score
    sCond_np = sphere.hkScore(cosTh, tau)*tangent
    
    #create and return torch tensors
    x0 = torch.from_numpy(x0_np).float().to(device)
    u = torch.from_numpy(u_np).float().to(device)
    x = torch.from_numpy(x_np).float().to(device)
    sCond = torch.from_numpy(sCond_np).float().to(device)
    
    return u, x, sCond

# TRAIN NET

def trainSmiley(
    scoreNet,
    optimizer,
    sampler,
    batchSize=6400,
    nSteps=10001,
    multTimes=True,
    uTest=.5,
    minU=0.,
    maxU=1.,
    device='cpu',
    printEvery=100
):
    """
    Train a smiley score network.

    sampler should be one of:
        sampleTrainingUniform
        sampleTrainingDirect
        sampleTraining
    """

    lossHistory = []

    scoreNet.train()

    for step in tqdm(range(nSteps)):

        # Generate training batch
        with torch.no_grad():
            u, x, target = sampler(
                batchSize,
                multTimes=multTimes,
                uTest=uTest,
                minU=minU,
                maxU=maxU,
                device=device
            )

        # Network prediction
        inp = torch.cat((x, u), dim=-1)
        F = scoreNet(inp)
        prediction = torch.cross(F, x, dim=-1)

        # Loss
        loss = torch.sum((prediction - target)**2, dim=-1).mean()

        # Gradient update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossHistory.append(loss.item())

        # Diagnostic
        if step % printEvery == 0:
            u_np = u.detach().cpu().numpy()
            tau_np = sde.tauFromU(u_np, t0, tF)
            x_np = x.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()

            s_exact = scoreSmiley(x_np, tau_np)

            lossFloor = np.sum(
                (target_np - s_exact)**2,
                axis=-1
            ).mean()

            print(
                f"step {step:6d}   "
                f"loss {(lossHistory[-1] - lossFloor):.6f}"
            )

    return lossHistory


# PLOTTING NEURAL NET OUTPUT

# Create uniform grid of unit vectors
nPhi=100
nZ=100

phi = np.linspace(0, 2*np.pi, nPhi, endpoint=False)
z = np.linspace(-1, 1, nZ)[:, None]
sinTh = np.sqrt(1 - z**2)

vX = np.cos(phi)*sinTh
vY = np.sin(phi)*sinTh
vZ = np.broadcast_to(z, (nZ, nPhi))
v = np.stack([vX, vY, vZ], axis=-1) 

vTorch = torch.from_numpy(v.reshape((10000,3))).float().to(device)

def plotScoreNormExact(uTest):
    """Plot norm of exact score field at non-batched time uTest"""

    tTest = sde.tauFromU(uTest, t0, tF)
    
    scoreDist = scoreSmiley(v, tTest)
    values = sphere.norm(scoreDist)
    
    plt.figure()
    plt.imshow(values, origin='lower', extent=[0, 2*np.pi, -1, 1])
    plt.colorbar(orientation='horizontal');

def plotScoreNormNet(scoreNet, uTest, plotGauge = False):
    """Plot norm of score field from neural net scoreNet at non-batched time uTest.
    If plotGauge, the component of neuralNet output that doesn't contribute to score is also plotted."""

    uTorch = torch.full((10000, 1), uTest, dtype=torch.float32, device=device)
    
    omega = scoreNet(torch.cat((vTorch, uTorch),dim=-1))
    scores = torch.cross(omega, vTorch, dim=-1)

    norm = torch.sqrt(torch.sum((scores)**2, dim=-1))
    norm_np = norm.reshape((100,100)).detach().cpu().numpy()

    plt.figure()
    plt.imshow(norm_np, origin='lower', extent=[0, 2*np.pi, -1, 1])
    plt.colorbar(orientation='horizontal');

    if plotGauge:
        gauge = torch.sum(omega*vTorch, dim=-1)
        gauge_np = gauge.reshape((100,100)).detach().cpu().numpy()

        plt.figure()
        plt.imshow(gauge_np, origin='lower', extent=[0, 2*np.pi, -1, 1])
        plt.colorbar(orientation='horizontal');

# ODE/SDE HELPER FUNCTIONS

def score_np(x, t, scoreNet):
    x = torch.from_numpy(x).float().to(device)
    u_np = sde.uFromTau(t, t0, tF)
    u = torch.full((x.shape[0], 1), u_np, dtype=torch.float32, device=device)
    
    inp = torch.cat((x, u), dim=-1)
    with torch.no_grad():
        F = scoreNet(inp)
        
    score = torch.cross(F, x, dim=-1)

    return score.detach().cpu().numpy()