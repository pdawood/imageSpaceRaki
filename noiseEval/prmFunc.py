
import numpy as np
import torch 

from cnn.utils_imgSpc.act2toelpitz import rakiImgRecoToel
from cnn.utils_imgSpc.fft import IFFT2

from cnn.rakiModels_imgSpc import rakiImgSpc
from grappa.grappaRecoImg import grappaImg


NREP = 1000

def prmRef(kspc_fs:torch.Tensor, sclFac:float, sigma:torch.Tensor, c:int=1)->torch.Tensor:
    '''
    Obtains standard deviation matrix of reference images using Monte Carlo simulations. 

    Parameters
    ----------
    kspc_fs : torch.Tensor
        Fully-sampled, multi-coil k-space
    sclFac : float
        k-space scaling factor
    sigma : torch.Tensor
            Coil correlation matrix
    c : int, optional
        Scaling noise standard deviation  The default is 1.

    Returns
    -------
    reps_i_std : torch.Tensor
        Standard deviation map of fully sampled, coil-combined reference image.

    '''
    nC, nP, nR = kspc_fs.shape 
    reps_i = torch.zeros((NREP, nP, nR), dtype=kspc_fs.dtype)    
    for jj in range(NREP):
        gn = torch.randn(kspc_fs.shape) + 1j*torch.randn(kspc_fs.shape)
        gn = torch.einsum('ijk,im->mjk',gn,sigma)
        kspc_fs_tmp = torch.clone(kspc_fs) + c*sclFac*gn
        reco_tmp = IFFT2(kspc_fs_tmp)
        reco_tmp_rss = torch.sum(torch.abs(reco_tmp)**2,axis=0)**0.5    
        reps_i[jj,:,:] = reco_tmp_rss
    reps_i = torch.abs(reps_i.detach())
    reps_i_mean = torch.mean(reps_i, dim=0)
    reps_i_std = torch.std(reps_i[:,:,:], dim=0)
    return reps_i_std

def prmToel(kspc_zf:torch.Tensor, imgW:dict, actW:dict, sclFac:float, kspc_fs:torch.Tensor, R:int, toelW:dict, sigma:torch.Tensor, c:int=1)->torch.Tensor:
    '''
    Computes RAKI g-factor maps using Monte Carlo simulations on image space
    reco with Toelpitz representation of actication masks.

    Parameters
    ----------
    kspc_zf : torch.Tensor
        Zero-filled, undersampled multi-coil k-space
    imgW : dict
        Image space weights ordered in dictionary
    actW : dict
        Actvation masks in image space ordered in dictionary
    sclFac : float
        k-space scaling factor
    kspc_fs : torch.Tensor
        Fully sampled, multi-coil k-space
    R : int
        Undersampling rate
    toelW : dict
        Activation masks in Toelpitz representation
    sigma : torch.Tensor
        Coil correlation matrix    
    c : int, optional
        Scaling noise standard deviation  The default is 1.

    Returns
    -------
    g : torch.Tensor
        RAKI g-factor by Monte Carlo and Toelpitz representation

    '''
    nC, nP, nR = kspc_zf.shape 
    reps_i = torch.zeros((NREP, nP, nR), dtype=kspc_zf.dtype)
    for jj in range(NREP):
        gn = torch.randn(kspc_zf.shape) + 1j*torch.randn(kspc_zf.shape)
        gn = torch.einsum('ijk,im->mjk',gn,sigma)
        kspc_zf_tmp = torch.clone(kspc_zf) + c*sclFac*gn     
        reco_tmp,_ = rakiImgRecoToel(IFFT2(kspc_zf_tmp), imgW, actW, toelW)
        reco_tmp_rss = torch.sum(torch.abs(reco_tmp)**2,axis=0)**0.5
        reps_i[jj, :, :] = reco_tmp_rss
    reps_i = torch.abs(reps_i.detach())
    reps_i *= 1/np.sqrt(nP*nR)
    reps_i_mean = torch.mean(reps_i, dim=0)
    reps_i_std = torch.std(reps_i[:,:,:], dim=0)
    reps_i_std_ref = prmRef(kspc_fs, sclFac, sigma, c)
    g = reps_i_std/reps_i_std_ref/np.sqrt(R)
    return g
     
def prmImgRaki(kspc_zf:torch.Tensor, imgW:dict, actW:dict, sclFac:float, kspc_fs:torch.Tensor, R:int, sigma:torch.Tensor, c:int=1)->torch.Tensor:
    '''
    Computes RAKI g-factor maps using Monte Carlo simulations on image space
    reco.     

    kspc_zf : torch.Tensor
        Zero-filled, undersampled multi-coil k-space
    imgW : dict
        Image space weights ordered in dictionary
    actW : dict
        Actvation masks in image space ordered in dictionary
    sclFac : float
        k-space scaling factor
    kspc_fs : torch.Tensor
        Fully sampled, multi-coil k-space
    R : int
        Undersampling rate
    sigma : torch.Tensor
        Coil correlation matrix    
    c : int, optional
        Scaling noise standard deviation  The default is 1.

    Returns
    -------
    g : torch.Tensor
        RAKI g-factor by Monte Carlo image space reco

    '''
    nC, nP, nR = kspc_zf.shape 
    reps_i = torch.zeros((NREP, nP, nR), dtype=kspc_zf.dtype)
    for jj in range(NREP): 
        gn = torch.randn(kspc_zf.shape) + 1j*torch.randn(kspc_zf.shape)
        gn = torch.einsum('ijk,im->mjk',gn,sigma)
        kspc_zf_tmp = torch.clone(kspc_zf) + c*sclFac*gn   
        reco_tmp = rakiImgSpc(IFFT2(kspc_zf_tmp),imgW, actW)     
        reco_tmp_rss = torch.sum(torch.abs(reco_tmp)**2,axis=0)**0.5
        reps_i[jj, :, :] = reco_tmp_rss
    reps_i = torch.abs(reps_i.detach())
    reps_i *= 1/np.sqrt(nP*nR)
    reps_i_mean = torch.mean(reps_i, dim=0)
    reps_i_std = torch.std(reps_i[:,:,:], dim=0)
    reps_i_std_ref = prmRef(kspc_fs, sclFac, sigma, c)
    g = reps_i_std/reps_i_std_ref/np.sqrt(R)
    return g    


def prmGrappaImg(kspc_zf:torch.Tensor, w_mat_img:torch.Tensor, sclFac:float, kspc_fs:torch.Tensor, R:int, sigma:torch.Tensor, c:int=1)->torch.Tensor:
    '''
    Computes GRAPPA g-factor maps using Monte Carlo simulations on image space
    reco.        

    Parameters
    ----------
    kspc_zf : torch.Tensor
        Zero-filled, undersampled multi-coil k-space
    w_mat : torch.Tensor
        GRAPPA kernel in image space
    sclFac : float
        k-space scaling factor
    kspc_fs : torch.Tensor
        Fully sampled, multi-coil k-space
    R : int
        Undersampling rate
    sigma : torch.Tensor
        Coil correlation matrix
    c : int, optional
        Scaling noise standard deviation  The default is 1.

    Returns
    -------
    g : torch.Tensor
        GRAPPA g-factor by Monte Carlo image space reco


    '''
    nC, nP, nR = kspc_zf.shape 
    reps_i = torch.zeros((NREP, nP, nR), dtype=kspc_zf.dtype)
    for jj in range(NREP):
        gn = torch.randn(kspc_zf.shape) + 1j*torch.randn(kspc_zf.shape)
        gn = torch.einsum('ijk,im->mjk',gn,sigma)
        kspc_zf_tmp = torch.clone(kspc_zf) + c*sclFac*gn     
        reco_tmp = grappaImg(IFFT2(torch.clone(kspc_zf_tmp)), w_mat_img)
        reco_tmp_rss = torch.sum(torch.abs(reco_tmp)**2,axis=0)**0.5
        reps_i[jj, :, :] = reco_tmp_rss
    reps_i = torch.abs(reps_i.detach())
    reps_i_mean = torch.mean(reps_i, dim=0)
    reps_i_std = torch.std(reps_i[:,:,:], dim=0)
    reps_i_std_ref = prmRef(kspc_fs, sclFac, sigma, c)
    g = reps_i_std/reps_i_std_ref/R
    return g    
