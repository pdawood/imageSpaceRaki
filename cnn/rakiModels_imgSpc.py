
import torch
import torch.nn.functional as F
import torch.nn as nn 

from cnn.utils.complexUtils import complex_conv2d
from .utils_imgSpc.imgWeights import ksp2imgWeights 
from .utils_imgSpc.actMasks import activationMasks

class complexConv(nn.Module):
    """ My Torch module for creation of complex neural network with CNN arch.    
    """
    def __init__(self, kernel_size, inC, outC):
        super(complexConv, self).__init__()
  
        self.conv1 = complex_conv2d(in_channels=inC,
                                    out_channels=outC,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=0,
                                    dilation=[1, 1],
                                    groups=inC,
                                    bias=False) 
    def forward(self, input_data:torch.Tensor)->torch.Tensor:           
           return self.conv1(input_data)    

def actImgSpc(input_tensor:torch.Tensor, kernel_tensor:torch.Tensor)->torch.Tensor:
    '''
    Performs activation in image space domain in one network layer.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Singal to be activated in image space.
    kernel_tensor : torch.Tensor
        Activation mask in image space.

    Returns
    -------
    output_tensor : torch.Tensor
        Activated signal in image space.  

    '''
    nCh = kernel_tensor.shape[1] 
    # Determine padding values for 'same' mode
    kernel_height, kernel_width = kernel_tensor.shape[-2], kernel_tensor.shape[-1]
    padding_height = (kernel_height) // 2
    padding_width = (kernel_width) // 2
    input_tensor = input_tensor[None, ...]
    # Pad the input tensor
    input_padded = F.pad(input_tensor, (padding_width-1, padding_width, padding_height-1, padding_height), mode='circular') 
    kernel_size = kernel_tensor.shape[-2], kernel_tensor.shape[-1]
    conv = complexConv(kernel_size, nCh, nCh)
    conv.conv1.real_filter.weight = torch.nn.parameter.Parameter(data=torch.transpose(kernel_tensor, 0, 1).real, requires_grad=False)
    conv.conv1.imag_filter.weight = torch.nn.parameter.Parameter(data=torch.transpose(kernel_tensor, 0, 1).imag, requires_grad=False)
    output_tensor = conv(input_padded)    
    return torch.squeeze(output_tensor)    

def getImgSpcParams(kspc_zf:torch.Tensor, net:torch.nn.Module, R:int)->tuple[dict,dict]:
    '''
    Function to obtain image space reconstruction parameters for RAKI inference 
    in image space domain. 

    Parameters
    ----------
    kspc_zf : torch.Tensor
       Zero-filled, undersampled multi-coil k-space
    net : torch.nn.Module
        Trained convolutional neural network.
    R : int
        Undersampling rate

    Returns
    -------
    tuple[dict,dict]
        [Image space weights, activation masks in image space domain]
        Ordered in dictionary, i.e. dict[i] containes weights of the i-th network layer.

    '''
    _, nP, nR = kspc_zf.shape
    imgW = ksp2imgWeights(net, R, [nP, nR])
    actW = activationMasks(kspc_zf, net)
    return (imgW, actW)

def rakiImgSpc(kspc_zf_fft:torch.Tensor, imgW:dict, actW:dict)->torch.Tensor:
    '''
    Performs RAKI image space inference.

    Parameters
    ----------
    kspc_zf_fft : torch.Tensor
        Zero-filled, undersampled multi-coil k-space.
    imgW : dict
        Image space weights
    actW : dict
        Activation masks in image space domain.

    Returns
    -------
    reco : torch.Tensor
        Multi-coil image reconstructions. 

    '''
    nLayers = len(imgW.keys())
    reco = torch.sum(kspc_zf_fft.repeat(imgW[1].shape[0],1,1,1) * imgW[1], 1)
    reco = actImgSpc(reco, actW[1])
    reco = torch.squeeze(reco)

    for jj in range(2,nLayers):
        reco = torch.sum(reco.repeat(imgW[jj].shape[0],1,1,1) * imgW[jj], 1)
        reco = actImgSpc(reco, actW[jj])
        reco = torch.squeeze(reco)
        
    reco = torch.sum(reco.repeat(imgW[nLayers].shape[0],1,1,1) * imgW[nLayers], 1)             
    return reco