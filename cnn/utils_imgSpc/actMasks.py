
import torch 
import torch.nn.functional as F

from .fft import IFFT2
from ..utils.complexUtils import cLeakyReLu 

def padX(tensor:torch.Tensor, imgSize:list[int])->torch.Tensor:
    '''
    Zero-pads tensor to desired size.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be padded
    imgSize : list[int]
        Desired image size

    Returns
    -------
    tensor_p : TYPE
        Zero-padded tensor.

    '''
    nP, nR = imgSize
    _, _, tensorSizePE, tensorSizeRO = tensor.shape
    padPE  = int((nP-tensorSizePE)//2)
    padRO = int((nR-tensorSizeRO)//2) 
    cP = 1 if (nP-tensorSizePE)%2==1 else 0
    cR = 1 if (nR-tensorSizeRO)%2==1 else 0
    tensor_p = F.pad(tensor, (padRO, padRO+cR, padPE, padPE+cP), 'constant', 0)
    return tensor_p

def getActMask(tensor0:torch.Tensor, a:float, imgSize:list[int])->torch.Tensor:
    '''
    Computation of the activation mask for given signal to be activated in k-space. 
    For details, see the theory-section of the paper. 

    Parameters
    ----------
    tensor0 : torch.Tensor
        Signal to be activated in k-space
    a : float
        The slope-parameter in the negative part of Leaky Rectifier Linear Unit
    imgSize : list[int]
        Image size

    Returns
    -------
    mask_cplx_fft : torch.Tensor
        The activation mask in the image space domain.

    '''
    tensor = padX(tensor0, imgSize)
    mask_real = torch.where(tensor.real >0, 1.0, a)
    mask_imag = torch.where(tensor.imag >0, 1.0, a)
    mask = mask_real + 1j* mask_imag
    output_act_mask = mask.real * tensor.real + 1j* mask.imag * tensor.imag 
    mask_cplx = (output_act_mask * torch.conj(tensor) ) * 1/((tensor.abs()**2)+1e-12)
    mask_cplx = torch.squeeze(mask_cplx)   
    mask_cplx_fft = IFFT2(mask_cplx)
    mask_cplx_fft = torch.flip(mask_cplx_fft,(-2,-1))
    return mask_cplx_fft[None, ...]

def activationMasks(kspc_zf:torch.Tensor, net:torch.nn.Module)->dict:
    '''
    Function to obtain the activation masks in image space domain from trained 
    interpolation network.

    Parameters
    ----------
    kspc_zf : torch.Tensor
        The zero-filled, undersampled multi-coil k-space.
    net : torch.nn.Module    
        The trained convolutional neural network.

    Returns
    -------
    dict
        The activation masks in image space domain in a dictionary, ordered 
        by the layer number, i.e. dict[i] contains activation masks of i-th network 
        layer.

    '''
    _, nP, nR = kspc_zf.shape
    actWeights = {} 
    # input layer
    #a= 0
    a = 0.5
    #a = 1
    x = net.conv1(kspc_zf[None, ...])
    actWeights[1] = getActMask(x.detach(), a, [nP, nR])
    x = cLeakyReLu(x, a)
    # hidden layer
    for numLayer in range(2, net.cnt):
        convLayer = getattr(net, 'conv' + str(numLayer))
        x = convLayer(x)
        actWeights[numLayer] = getActMask(x.detach(), a, [nP, nR])
        x = cLeakyReLu(x, a)
    return actWeights