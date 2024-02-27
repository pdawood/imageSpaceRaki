
import torch
import torch.nn as nn

from torch.nn import  Module

class complex_conv2d(Module):
    '''
    Custom PyTorch 2D Convolution Layer to perform complex-valued convolutions. 
    '''    
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias):

        super(complex_conv2d, self).__init__()
        
        self.real_filter = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride,
                                     padding,
                                     dilation,
                                     groups,
                                     bias) 
        
        self.imag_filter = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride,
                                     padding,
                                     dilation,
                                     groups,
                                     bias) 


    def forward(self, complex_input:torch.Tensor)->torch.Tensor:
        real_input = complex_input.real
        imag_input = complex_input.imag

        real_filter_real_input = self.real_filter(real_input)
        real_filter_imag_input = self.real_filter(imag_input)
        imag_filter_real_input = self.imag_filter(real_input)
        imag_filter_imag_input = self.imag_filter(imag_input)

        real_output = real_filter_real_input - imag_filter_imag_input
        imag_output = real_filter_imag_input + imag_filter_real_input

        complex_output = torch.complex(real_output, imag_output)

        return complex_output 




def cLeakyReLu(complex_input:torch.Tensor, a:float)->torch.Tensor:
    '''
    Custom activation function based on Leaky Rectifier Unit to 
    perform complex-valued convolutions. 

    Parameters
    ----------
    complex_input : torch.Tensor
        Signal to be activated in k-space.
    a : float
        Slope-parameter 

    Returns
    -------
    complex_output : torch.Tensor
        Activated signal

    '''
    negative_slope = a
    real_input = complex_input.real
    imag_input = complex_input.imag 
    m = nn.LeakyReLU(negative_slope)

    m_real = m(real_input)
    m_imag = m(imag_input)

    complex_output = torch.complex(m_real, m_imag)

    return complex_output