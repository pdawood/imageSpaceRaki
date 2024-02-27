
import torch
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import kstest, norm

PTHRES = 0.05

def kolSmirTest(reps:torch.Tensor, sclFac:float)->None:
    '''
    Kolmogorov-Smirnov Test for normal distribution, applied to Monte Carlo
    simulations on RAKI reconstructions.

    Parameters
    ----------
    reps : torch.Tensor
        The RAKI/GRAPPA pseudo-replicas from Monte Carlo simulations
    sclFac : float
        k-space scaling factor

    Returns
    -------
    None


    '''
    reps = reps.detach().numpy()
    _, nP, nR = reps.shape 
    pMap = np.zeros((nP, nR))
    sMap = np.zeros((nP, nR))
    for ii in range(nP):
        for jj in range(nR):
            data = reps[:,ii,jj]
            loc, scale = norm.fit(data)   
            n = norm(loc=loc, scale=scale)        
            sMap[ii,jj], pMap[ii,jj] = kstest(data, n.cdf)    
    pMapThres = np.where(pMap>PTHRES, 1, 0)            
    fig, axs = plt.subplots(2,1,figsize=[12,20], dpi=200, constrained_layout=True)
    pcm0 = axs[0].imshow(np.rot90(pMap),vmin=0, vmax=1)
    pcm1 = axs[1].imshow(np.rot90(pMapThres), vmin=0, vmax=1)
    cb0 = fig.colorbar(pcm0, ax=axs[0], shrink=1, ticks=[0,1])
    cb0.ax.tick_params(labelsize=36)
    cb1 = fig.colorbar(pcm1, ax=axs[1], shrink=1, ticks=[0,1])
    cb1.ax.tick_params(labelsize=36)
    for jj in range(2):
        axs[jj].axis('off')


def printKSTest(reps:torch.Tensor, sclFac:float)->None:
    '''
    Print-Function for Kolmogorov Smirnov test for normal distribution.

    Parameters
    ----------
    reps : torch.Tensor
        The RAKI/GRAPPA pseudo-replicas from Monte Carlo simulations
    sclFac : float
        k-space scaling factor

    Returns
    -------
    None

    '''
    reps_ = np.rot90(reps, 1, (1,2))
    reps_ *= 1/sclFac
    stdMap = np.std(reps_,0)
    maxStd = 10
    font = {'weight' : 'bold',
            'size'   : 26}
    matplotlib.rc('font', **font)
    #headm
    points = [[18,25],[22,42],[30,33],[33,15]]
    lbl = ['A', 'B', 'C', 'D'] 
    #font = {'fontsize':36, 'color': 'k', 'fontweight': 'bold'}
    fig, axs = plt.subplots(1,1, figsize=[8,8],dpi=100)
    pcm0 = axs.imshow(stdMap,vmin=0, vmax=maxStd)
    axs.axis('off')    
    cb0 = fig.colorbar(pcm0, ax=axs, shrink=0.8, ticks=np.arange(0,maxStd+1,2))
    cb0.ax.tick_params(labelsize=36)
    fig1, axsP = plt.subplots(2,2, figsize=[12,12], dpi=200)
    for e,p in enumerate(points):
        ii, jj = p[0], p[1]
        lbl_tmp = lbl[e]
        data = reps[:,ii,jj]
        loc, scale = norm.fit(data)   
        n = norm(loc=loc, scale=scale)         
        axs.text(ii,jj, lbl_tmp, fontdict=font)
        axs_tmp = axsP[e//2,e%2]
        axs_tmp.hist(data, density=True, bins=20,label='Monte Carlo')
        x = np.arange(data.min(), data.max()+0.2, 0.2)
        axs_tmp.plot(x, n.pdf(x), label='Normal Fit', linewidth=4)
        axs_tmp.set_xlabel('Pixel Magnitude', fontdict=font)
        axs_tmp.set_ylabel('Probability Density', fontdict=font)
        axs_tmp.set_xlim([loc-20,loc+20])
        axs_tmp.set_ylim([0,0.2])
        axs_tmp.set_xticks([loc-20,loc, loc+20])
        if e==0:
            axs_tmp.legend(fontsize=22)
        axs_tmp.set_title(lbl_tmp, fontdict=font)
        axs_tmp.grid()
    plt.tight_layout()
    plt.show()