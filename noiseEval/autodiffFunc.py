
import numpy as np
import torch 

from cnn.utils_imgSpc.act2toelpitz import rakiImgRecoToel
from cnn.utils_imgSpc.fft import IFFT2

from grappa.grappaRecoImg import grappaImg

def autoDiffToel(kspc_zf:torch.Tensor, imgW:dict, actW:dict, toelW:dict, p:torch.Tensor, R:int, sigma:torch.Tensor)->torch.Tensor:
    '''
    Computes RAKI g-factor maps using auto-differentiation
    

    Parameters
    ----------
    kspc_zf : torch.Tensor
        Zero-filled, undersampled multi-coil k-space
    imgW : dict
        Image-space convolution weights for each network layer
    actW : dict
        Activation weights in image space for each network layer
    toelW : dict
        Activation weights in Toelpitz representation
    p : torch.Tensor
        Coil-combination weights
    R : int
        Undersapling rate
    sigma : torch.Tensor
        Coil correlation matrix

    Returns
    -------
    g : torch.Tensor
        RAKI g-factor by auto-differentiation

    '''
    nC, nP, nR = kspc_zf.shape
    kspc_zf_fft = IFFT2(kspc_zf)
    kspc_zf_fft.requires_grad_(True)  
    #imgW[2] *= 1/np.sqrt(nP*nR)
    img_reco,_ = rakiImgRecoToel(kspc_zf_fft, imgW, actW, toelW)
    img_reco_rss = torch.sum(torch.abs(img_reco)**2,axis=0)**0.5
    stdMap = torch.zeros(nP,nR)
    for xx in range(0,nP):
        for yy in range(0,nR):
            torch.abs(img_reco_rss[xx,yy]*(1/np.sqrt(nP*nR))).backward(retain_graph=True)
            J = kspc_zf_fft.grad            
            J = torch.einsum('ijk,im->mjk',J, sigma)
            J = J.flatten()
            std_gradient = np.sqrt(torch.matmul(J, torch.conj(J).t()).real)            
            stdMap[xx,yy] = std_gradient
            kspc_zf_fft.grad.data.zero_() 
    p_sigma = torch.einsum('ij,ik->kj',p,sigma)
    stdMapRef = torch.sqrt(torch.matmul(p_sigma.T, torch.conj(p_sigma)))  
    stdMapRef = torch.diag(stdMapRef).reshape(nP,nR)
    g = stdMap/stdMapRef/np.sqrt(R)
    return g.real

def autoDiffGrappa(kspc_zf:torch.Tensor, w_im:torch.Tensor, p:torch.Tensor, R:int, sigma:torch.Tensor)->torch.Tensor:
    '''
    Computes GRAPPA g-factor maps using auto-differentiation
        

    Parameters
    ----------
    kspc_zf : torch.Tensor
        Zero-filled, undersampled multicoil k-space
    w_im : torch.Tensor
        GRAPPA image space weights
    p : torch.Tensor
        Coilcombination weights
    R : int
        Undersapling factor
    sigma : torch.Tensor
        Coil correlation matrix

    Returns
    -------
    g : torch.Tensor
        GRAPPA g-factor by auto-differentiation

    '''
    _, nP, nR = kspc_zf.shape
    kspc_zf_fft = IFFT2(kspc_zf)
    kspc_zf_fft.requires_grad_(True)  
    img_reco = grappaImg(kspc_zf_fft, w_im)
    img_reco_rss = torch.sum(torch.abs(img_reco)**2,axis=0)**0.5
    stdMap = torch.zeros(nP,nR)
    for xx in range(0,nP):
        for yy in range(0,nR):
            torch.abs(img_reco_rss[xx,yy]).backward(retain_graph=True)
            J = kspc_zf_fft.grad            
            J = torch.einsum('ijk,im->mjk',J, sigma)
            J = J.flatten()
            std_gradient = np.sqrt(torch.matmul(J, torch.conj(J).t()).real)            
            stdMap[xx,yy] = std_gradient
            kspc_zf_fft.grad.data.zero_() 
    p_sigma = torch.einsum('ij,ik->kj',p,sigma)
    stdMapRef = torch.sqrt(torch.matmul(p_sigma.T, torch.conj(p_sigma)))  
    stdMapRef = torch.diag(stdMapRef).reshape(nP,nR)
    g = stdMap/stdMapRef/R    
    return g.real
    