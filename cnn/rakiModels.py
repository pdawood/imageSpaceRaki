
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange
from .utils.complexCNN import complexNet
from .utils.shapeAnalysis import extractDatCNN, fillDatCNN

RAKI_RECO_DEFAULT_LR = 0.0005
RAKI_RECO_DEFAULT_EPOCHS = 500

IRAKI_RECO_DEFAULT_INIT_LR = 5e-4
IRAKI_RECO_DEFAULT_LR_DECAY = {
    4 : 3e-5,
    5 : 4e-5,
    6 : 6e-5
}
IRAKI_RECO_DEFAULT_ACS_NUM = 65

def trainRaki(acs:torch.Tensor, R:int, layer_design:dict)->torch.nn.Module:
    '''
    Trains RAKI in k-space.

    Parameters
    ----------
    acs : torch.Tensor
        Auto-Calibration Signal.
    R : int
        DESCRIPTION.
    layer_design : dict
        Description of the network architecture. Here is a example with two hidden layers:
        
        layer_design_raki = {'num_hid_layer': 2, # number of hidden layers, in this case, its 2
                        'input_unit': nC,    # number channels in input layer, nC is coil number 
                            1:[256,(2,5)],   # the first hidden layer has 256 channels, and a kernel size of (2,5) in PE- and RO-direction
                            2:[128,(1,1)],   # the second hidden layer has 128 channels, and a kernel size of (1,1) in PE- and RO-direction
                        'output_unit':[(R-1)*nC,(1,5)] # the output layer has (R-1)*nC channels, and a kernel size of (1,5) in PE- and RO-direction
                        }

    Returns
    -------
    net : torch.nn.Module
        Trained RAKI network.

    '''
    # Get Source- and Target Signals
    prc_data = extractDatCNN(acs,
                             R=R,
                             num_hid_layer=layer_design['num_hid_layer'],
                             layer_design=layer_design)
    
    trg_kspc = prc_data['trg'].permute((0, 3, 1, 2))
    src_kspc = prc_data['src'].permute((0, 3, 1, 2))

    net = complexNet(layer_design, R=R)    
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(net.parameters(), lr=RAKI_RECO_DEFAULT_LR) 
      
    for _ in trange(RAKI_RECO_DEFAULT_EPOCHS):    
        optimizer.zero_grad()
        pred_kspc = net(src_kspc)['tot']
        loss =   (criterion(pred_kspc.real, trg_kspc.real)
                + criterion(pred_kspc.imag, trg_kspc.imag)) 
        loss.backward()
        optimizer.step()   
    
    return net
    

def inferRaki(kspc_zf:torch.Tensor, R:int, layer_design:dict, net:torch.nn.Module)->torch.Tensor:
    '''
    Peforms RAKI inference in k-space.

    Parameters
    ----------
    kspc_zf : torch.Tensor
        Zero-filled, undersampled multi-coil k-space.
    R : int
        Undersampling rate
    layer_design : dict
        Description of the network architecture. Here is a example with two hidden layers:
        
        layer_design_raki = {'num_hid_layer': 2, # number of hidden layers, in this case, its 2
                        'input_unit': nC,    # number channels in input layer, nC is coil number 
                            1:[256,(2,5)],   # the first hidden layer has 256 channels, and a kernel size of (2,5) in PE- and RO-direction
                            2:[128,(1,1)],   # the second hidden layer has 128 channels, and a kernel size of (1,1) in PE- and RO-direction
                        'output_unit':[(R-1)*nC,(1,5)] # the output layer has (R-1)*nC channels, and a kernel size of (1,5) in PE- and RO-direction
                        }
    net : torch.nn.Module
        Trained RAKI network.

    Returns
    -------
    network_reco : torch.Tensor
        Multi-coil image reconstruction in k-space.

    '''

    kspc_zf_input = torch.unsqueeze(kspc_zf, 0)    
    #kspc_zf_input = torch.from_numpy(kspc_zf_input).type(torch.complex64)    
    # Estimate missing signals 
    kspc_pred = net(kspc_zf_input)['tot']
    #kspc_pred = kspc_pred.detach().numpy()
    kspc_pred = kspc_pred.permute((0, 2, 3, 1))
    kspc_pred = torch.squeeze(kspc_pred)

    # Put estimated signals bach into zero-filled kspace 
    network_reco = fillDatCNN(kspc_zf,
                              kspc_pred,
                              R,
                              num_hid_layer=layer_design['num_hid_layer'],
                              layer_design=layer_design)
    print('Finished Standard RAKI...')
    return network_reco    


def rakiReco(kspc_zf:torch.Tensor, acs:torch.Tensor, R:int, layer_design:dict)->(torch.Tensor, torch.nn.Module):
    '''
    This function trains RAKI, and puts the interpolated signals 
    into zero-filled k-space.
    
    Parameters
    ----------
    kspc_zf : torch.Tensor
        Zero-filled k-space, not including acs, in shape [coils, PE, RO].
    acs : torch.Tensor
        Auto-Calibration-Signal, in shape [coils, PE, RO].
    R : int
        Undersampling-Factor.
    layer_design : dict
        Description of the network architecture. Here is a example with two hidden layers:

            layer_design_raki = {'num_hid_layer': 2, # number of hidden layers, in this case, its 2
                        'input_unit': nC,    # number channels in input layer, nC is coil number 
                            1:[256,(2,5)],   # the first hidden layer has 256 channels, and a kernel size of (2,5) in PE- and RO-direction
                            2:[128,(1,1)],   # the second hidden layer has 128 channels, and a kernel size of (1,1) in PE- and RO-direction
                        'output_unit':[(R-1)*nC,(1,5)] # the output layer has (R-1)*nC channels, and a kernel size of (1,5) in PE- and RO-direction
                        }
    
    Returns
    -------

        network_reco : torch.Tensor
            RAKI k-space reconstruction, in shape [coils, PE, RO].
    '''
    print('Starting Standard RAKI...')
    # Get Source- and Target Signals
    prc_data = extractDatCNN(acs,
                             R=R,
                             num_hid_layer=layer_design['num_hid_layer'],
                             layer_design=layer_design)
    
    trg_kspc = prc_data['trg'].transpose((0, 3, 1, 2))
    src_kspc = prc_data['src'].transpose((0, 3, 1, 2))

    src_kspc = torch.from_numpy(src_kspc).type(torch.complex64)
    trg_kspc = torch.from_numpy(trg_kspc).type(torch.complex64)
    
    net = complexNet(layer_design, R=R)    
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(net.parameters(), lr=RAKI_RECO_DEFAULT_LR) 
      
    for _ in trange(RAKI_RECO_DEFAULT_EPOCHS):    
        optimizer.zero_grad()
        pred_kspc = net(src_kspc)['tot']
        loss =   (criterion(pred_kspc.real, trg_kspc.real)
                + criterion(pred_kspc.imag, trg_kspc.imag)) 
        loss.backward()
        optimizer.step()

    kspc_zf_input = np.expand_dims(kspc_zf, axis=0)    
    kspc_zf_input = torch.from_numpy(kspc_zf_input).type(torch.complex64)    
    # Estimate missing signals 
    kspc_pred = net(kspc_zf_input)['tot']
    kspc_pred = kspc_pred.detach().numpy()
    kspc_pred = kspc_pred.transpose((0, 2, 3, 1))
    kspc_pred = np.squeeze(kspc_pred)

    # Put estimated signals bach into zero-filled kspace 
    network_reco = fillDatCNN(kspc_zf,
                              kspc_pred,
                              R,
                              num_hid_layer=layer_design['num_hid_layer'],
                              layer_design=layer_design)
    print('Finished Standard RAKI...')
    return network_reco, net


