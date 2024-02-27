
import numpy as np
import torch 
import torch.nn.functional as F

from .fft import IFFT2

def getNetWeights(net:torch.nn.Module)->dict:
    '''
    Function to extract the convolution filter weights of CNN.

    Parameters
    ----------
    net : torch.nn.Module
        Trained convolutional neural network

    Returns
    -------
    dict
        Dictionary containing the complex-valued convolution filters detached from the autodiff graph.

    '''
    netWeights = {}
    for jj in range(1,net.cnt+1):
        convLayer = getattr(net, 'conv' + str(jj))
        netWeights[jj] = convLayer.real_filter.weight.detach() + 1j*convLayer.imag_filter.weight.detach()
    return netWeights

def padFirstKernel(kernel:torch.Tensor, R:int)->torch.Tensor:
    '''
    Function to zero-pad the first convolution filter in the network. 
    This is necessary because it is dilated with factor R in the network in training.

    Parameters
    ----------
    kernel : torch.Tensor
        Convolution filter of the first network layer
    R : int
        Undersampling rate 

    Returns
    -------
    kernel_embed : TYPE
        The zero-padded convolution filter.

    '''
    layerOut, layerIn, kernelSizePE, kernelSizeRO = kernel.shape
    dilationEmbed = kernelSizePE + (kernelSizePE-1)*(R-1)
    kernel_embed = torch.zeros((layerOut, layerIn, dilationEmbed, kernelSizeRO),dtype=kernel.dtype, requires_grad=False)
    kernel_embed[:,:,::R,:] = kernel
    return kernel_embed

def getImgWeights(kernel:torch.Tensor, imgSize:list[int])->torch.Tensor:
    '''
    Transform k-space convolution filters into image space filters

    Parameters
    ----------
    kernel : torch.Tensor
        Convolution filter in k-space
    imgSize : list[int]
        Image size

    Returns
    -------
    kernel_fft : TYPE
        Convolution filter in the image space domain.

    '''
    nP, nR = imgSize
    _, _, kernelSizePE, kernelSizeRO = kernel.shape
    padding_height = int((nP-kernelSizePE)//2)
    padding_width = int((nR-kernelSizeRO)//2)
    cP = 1 if (nP-kernelSizePE)%2==1 else 0
    cR = 1 if (nR-kernelSizeRO)%2==1 else 0
    kernel_padded = F.pad(torch.flip(kernel,(-2,-1)), (padding_width+cR, padding_width, padding_height+cP, padding_height), mode='constant', value=0) 
    kernel_fft = np.sqrt(nP*nR)*IFFT2(kernel_padded)    
    return kernel_fft

def getImgWeightsFinal(kernel:torch.Tensor, R:int, imgSize:list[int]):
    '''
    Transform the k-space convolution filter of the last network layer 
    according to: Breuer F, et al. MRM. 2009;62(3): 739.46.

    Parameters
    ----------
    kernel : torch.Tensor
        Convolution filter in k-space
    R : int
        Undersampling rate
    imgSize : list[int]
        Image size

    Returns
    -------
    kernel_fft : TYPE
        Convolution filter in image space domain.

    '''
    kernel = torch.flip(kernel, (-2,-1))
    ny, nx = imgSize
    nOut_int = kernel.shape[0]
    nIn = kernel.shape[1]
    kernelSizePE = kernel.shape[2]
    kernelSizeRO = kernel.shape[3]
    nOut = int(nOut_int / (R))
    # Perform Fourier transforms of the image and kernel
    kernel = torch.reshape(kernel, (nOut, R, nIn, kernelSizePE, kernelSizeRO ))
    tmpDimPE = (R)*kernelSizePE 
    tmpDimRO = kernelSizeRO
    kernelInt = torch.zeros((nOut, nIn, tmpDimPE, tmpDimRO ),dtype=kernel.dtype)

    for hh in range(R): 
        kernelInt[:,:,hh:hh+1,:] = kernel[:,hh,:,:,:]
            
    W_k = torch.zeros((nOut, nIn, ny, nx), dtype=kernel.dtype)
    W_k[:, :, int((ny/R-kernelSizePE)*R/2):int((ny/R+kernelSizePE)*R/2),int(nx/2 - kernelSizeRO/2):int(nx/2 + kernelSizeRO/2)] = kernelInt 
    kernel_fft = IFFT2(W_k)
    return kernel_fft

def ksp2imgWeights(net:torch.nn.Module, R:int, imgSize:list[int])->dict: 
    '''
    Transforms k-space convolution filters of (trained) network into image space domain

    Parameters
    ----------
    net : torch.nn.Module
        Trained convolutional neural network.
    R : int
        Undersampling rate
    imgSize : list[int]
        Image size

    Returns
    -------
    dict
        The convolution filters in the image space domain in a dictionary, ordered 
        by the layer number, i.e. dict[i] contains image space weights of i-th network 
        layer.

    '''
    imgWeights = {}
    netWeights = getNetWeights(net)
    netWeights[1] = padFirstKernel(netWeights[1], R)
    nLayers = len(netWeights.keys()) 

    for jj in range(1, nLayers):
        imgWeights[jj] = getImgWeights(netWeights[jj], imgSize) 
        
    imgWeights[nLayers] = getImgWeightsFinal(netWeights[nLayers], R, imgSize)
        
    return imgWeights
