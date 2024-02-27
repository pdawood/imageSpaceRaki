
import numpy as np
import torch 
import torch.nn.functional as F
from .fft import IFFT2

def getToelpitzMat(actMask: torch.Tensor)->torch.Tensor:
    '''
    Transforms convolution with kernel into matrix-multiplication with toelpitz-representation 
    of kernel. 

    Parameters
    ----------
    actMask : torch.Tensor
        Convolution kernel.

    Returns
    -------
    actMaskToel : TYPE
        Toelpitz-representation of kernel.

    '''
   
    _, nCh, nP, nR = actMask.shape
    # Determine padding values for 'same' mode
    kernel_height, kernel_width = actMask.shape[-2], actMask.shape[-1]
    padding_height = (kernel_height -1 ) // 2
    padding_width = (kernel_width - 1) // 2

    # Pad the input tensor
    idxTensor = torch.arange(nP*nR).reshape(nP, nR)
    idxTensor = idxTensor[None, None, ...]
    idxTensorPadded = F.pad(idxTensor, (padding_width, padding_width, padding_height, padding_height), mode='circular')
    idxTensorPadded = torch.squeeze(idxTensorPadded)
    idxTensorPadded = idxTensorPadded[:nP,:nR]
    #idxTensorPadded = torch.roll(idxTensorPadded, (-1,-1), (0,1))
    idxTensorPadded = idxTensorPadded.detach().numpy().reshape(nP*nR)
    actMaskToel = torch.zeros((nCh, nP, nR, nP*nR), dtype=actMask.dtype)
    actMask = actMask.reshape(nCh, nP*nR)
    for bb in range(nP*nR):
        actMaskToel[:,0,0,bb] = actMask[:,int(np.argwhere(idxTensorPadded==bb))]
    for jj in range(0,nP):
        for kk in range(0,nR):
            actMaskToel[:,jj, kk, :] = torch.roll(actMaskToel[:,0,0,:],((jj*nR+kk)),1)    
    actMaskToel = actMaskToel.reshape(nCh, (nP)*(nR), nP*nR)
    return actMaskToel


def actMask2toelpitz(actWeights:dict)->dict:
    '''
    Transforms activation masks into Toelpitz-representation.

    Parameters
    ----------
    actWeights : dict
        Dictionary of activation masks.

    Returns
    -------
    dict
        Dictionary of activation masks in Toelpitz-representation.

    '''
    nMaps = len(actWeights.keys())
    toelWeights = {}    
    for jj in range(1,nMaps+1):
        toelWeights[jj] = getToelpitzMat(actWeights[jj])    
    return toelWeights
    
 
def rakiImgRecoToel(kspc_zf_fft:torch.Tensor, imgW:dict, actW:dict, toelWeights:dict=None)->tuple[torch.Tensor, dict]:    
    '''
    Performs RAKI image space domain reco using Toelpitz-representation of activation masks.    

    Parameters
    ----------
    kspc_zf_fft : torch.Tensor
        Zero-filled, undersampled multi-coil k-space
    imgW : dict
        Image space weights in dictionary.
    actW : dict
        Activation masks in dictionary
    toelWeights : dict, optional
        Toelpitz-representation of activation masks, if computed. The default is None.

    Returns
    -------
    tuple [reco : torch.Tensor, toelWeights : dict]
        [Multi-coil image reconstructions, Activation masks in Toelpitz-representation] 

    '''
    nC, nP, nR = kspc_zf_fft.shape
    nLayers = len(imgW.keys())
    if toelWeights == None:
        toelWeights = actMask2toelpitz(actW)
    reco = torch.reshape(kspc_zf_fft, (nC, nR*nP))
    
    for jj in range(1,nLayers):
        nOut, nIn, _, _ = imgW[jj].shape
        reco = torch.einsum('ijk,jk->ik',torch.reshape(imgW[jj], (nOut, nIn, nP*nR)),reco)
        reco = torch.einsum('ijk,ik->ij',toelWeights[jj], reco)
    
    nOut, nIn, _, _ = imgW[nLayers].shape
    reco = torch.einsum('ijk,jk->ik',torch.reshape(imgW[nLayers], (nOut, nIn, nP*nR)),reco)
    reco = torch.reshape(reco, (nC, nP, nR))
    return (reco, toelWeights)
