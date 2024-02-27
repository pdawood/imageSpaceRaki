
import torch
import numpy as np
from cnn.utils_imgSpc.act2toelpitz import actMask2toelpitz

def rakiGfactorCalc(imgW:dict, actW:dict, p:torch.Tensor, R:int, sigma=torch.Tensor)->torch.Tensor:
    '''
    Analytical RAKI g-factor calculation.    

    Parameters
    ----------
    imgW : dict
        RAKI image space weights ordered in dictionary. 
    actW : dict
        RAKI activation masks in image space ordered in dictionary.
    p : torch.Tensor
        Coil-combination weights
    R : int
        Undersampling rate
    sigma : torch.Tensor
        Coil correlation matrix

    Returns
    -------
    g : torch.Tensor
        Analytical RAKI g-factor

    '''
    nLayers = len(imgW.keys())
    toelWeights = actMask2toelpitz(actW)
    nOut, nIn, nP, nR = imgW[1].shape
    J = torch.einsum('ijk,irk->ijrk',imgW[1].reshape(nOut,nIn,nP*nR), toelWeights[1])    
    for jj in range(2,nLayers):
        nOut, nIn, _, _ = imgW[jj].shape
        J = torch.einsum('ijk,jrko->irko',(1/np.sqrt(nP*nR))*imgW[jj].reshape(nOut,nIn,nP*nR),J)
        J = torch.einsum('ijkl,ihk->ijhl',J, toelWeights[jj])     
    nOut, nIn, _, _ = imgW[nLayers].shape    
    J = torch.einsum('ijk,jrko->irko',imgW[nLayers].reshape(nOut,nIn,nP*nR),J)       
    Jint = torch.einsum('ij,jhim->jhim',p.T,J)
    Jint = torch.sum(Jint, dim=0)

    Jint = Jint.permute(1,0,2).reshape(nP*nR,nP*nR,nOut)
    Jint = torch.einsum('ijk,km->ijm', Jint, sigma)
    Jint = Jint.reshape(nP*nR, -1)
    stdMap = torch.sqrt(torch.matmul(Jint, torch.conj(Jint).T).real)
    stdMap = torch.diag(stdMap).reshape(nP,nR)    
    
    p_sigma = torch.einsum('ij,ik->kj',p,sigma)
    stdMapRef = torch.sqrt(torch.matmul(p_sigma.T, torch.conj(p_sigma)))  
    stdMapRef = torch.diag(stdMapRef).reshape(nP,nR)
    g = stdMap/stdMapRef/np.sqrt(R)
    return g.real


def grappaGfactorCalc(w_im:dict, p:torch.Tensor, R:int, imgSize:list, sigma:torch.Tensor)->torch.Tensor:
    '''
    Analytical GRAPPA g-factor calculation.    

    Parameters
    ----------
    w_im : dict
        GRAPPA kernel in image space
    p : torch.Tensor
        Coil-combination weights
    R : int
        Undersampling rate
    imgSize : list
        Image size in shape [Phase, Readout]
    sigma : torch.Tensor
        Coil correlation matrix

    Returns
    -------
    g : torch.Tensor
        Analytical GRAPPA g-factor

    '''
    nC, nP, nR = imgSize
    w_im = w_im.reshape((nC, nC, nP*nR))
    Jint = torch.einsum('ij,ihj->hj',p,w_im)
    Jint = torch.einsum('ij,ik->kj', Jint, sigma)
    stdMap = torch.sqrt(torch.matmul(Jint.T, torch.conj(Jint)))
    stdMap = torch.diag(stdMap).reshape(nP, nR)
    
    p_sigma = torch.einsum('ij,ik->kj',p,sigma)
    stdMapRef = torch.sqrt(torch.matmul(p_sigma.T, torch.conj(p_sigma)))
    stdMapRef = torch.diag(stdMapRef).reshape(nP,nR)
    g = stdMap/stdMapRef/R
    return g.real