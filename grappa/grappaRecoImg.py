
import numpy as np
import torch 

#from ..cnn.utils_imgSpc.fft import IFFT2

def IFFT2(tensor:torch.Tensor)->torch.Tensor:
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(tensor,[-1,-2]),norm="ortho"),[-1,-2]) 

def kspc2ImgWeights(w_mat:torch.Tensor, kernel_grappa:dict, R:int, imgSize:list[int])->torch.Tensor:
    '''
    Transform GRAPPA kernel into image space domain
    according to: Breuer F, et al. MRM. 2009;62(3): 739.46.    

    Parameters
    ----------
    w_mat : torch.Tensor
        GRAPPA kernel
    kernel_grappa : dict
        Dictionary with keys 'phase' and 'read' for 
        GRAPPA kernel extension in PE- and RO-direction, respectively (e.g. kernel_design={'phase': 2, 'read':5}).
    R : int
        Undersampling rate
    imgSize : list[int]
        Image size in shape [Phase, Readout]

    Returns
    -------
    W_im : torch.Tensor
        GRAPPA kernel in image space

    '''
    nC, nP, nR = imgSize[0], imgSize[1], imgSize[2] 
    k_p, k_r = kernel_grappa['phase'],kernel_grappa['read'] 
    w_mat = w_mat.T
    w_mat = torch.reshape(w_mat, (nC, R, nC, k_p, k_r))
    w_mat = torch.flip(w_mat, (-2, -1))
    ws_kernel = torch.zeros((nC, nC, k_p*R, k_r), dtype=w_mat.dtype, requires_grad=False)
    for kk in range(R):
        ws_kernel[:, :, kk:k_p*R:R, :] = w_mat[:, kk, :, :, :]
    # zero padding of convolution kernel
    W_k = torch.zeros((nC, nC, nP, nR), dtype=w_mat.dtype)
    W_k[:, :, int((nP/R-k_p)*R/2):int((nP/R+k_p)*R/2),int(nR/2 - k_r/2):int(nR/2 + k_r/2)] = ws_kernel
    # inverse Fourier transform (convolution kernel --> image space)
    W_im = np.sqrt(nP*nR)*IFFT2(W_k)
    return W_im

def grappaImg(kspc_zf_fft:torch.Tensor, w_im:torch.Tensor)->torch.Tensor:
    '''
    Performs GRAPPA reco in image space domain

    Parameters
    ----------
    kspc_zf_fft : torch.Tensor
        FFT zero-filled, undersampled multi-coil k-spaces
    w_im : torch.Tensor
        GRAPPA kernel in image domain

    Returns
    -------
    recoImg : TYPE
        GRAPPA multi-coil image space reco 

    '''
    recoImg = torch.einsum('ijk,mijk->mjk', kspc_zf_fft, w_im)
    return recoImg