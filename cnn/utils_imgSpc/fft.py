
import torch 

def IFFT2(tensor:torch.Tensor)->torch.Tensor:
    '''
    Performs inverse Fast Fourier Transformation. 

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be inverse Fourier transformed

    Returns
    -------
    tensor: torch.Tensor
        Inverse Fourier transformed tensor

    '''
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(tensor,[-1,-2]),norm="ortho"),[-1,-2]) 

def FFT2(tensor:torch.Tensor)->torch.Tensor:
    '''
    Performs Fast Fourier Transformation. 

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be Fourier transformed

    Returns
    -------
    tensor: torch.Tensor
        Fourier transformed tensor

    '''
    return torch.fft.ifftshift(torch.fft.fft2(torch.fft.ifftshift(tensor,[-1,-2]),norm="ortho"),[-1,-2]) 
