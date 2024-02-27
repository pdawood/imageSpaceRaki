
import torch 
import numpy as np
import time

def extractTrainingDatGrap(acs:torch.Tensor, R:int, Nk_p:int, Nk_r:int)->dict:    
    '''
    Function to get source- and target-matrix from acs to calculate
    GRAPPA kernel. Source-matrix from zero-filled k-space is also generated.
    
    Parameters
    ----------
    acs : torch.Tensor
        Auto-Calibration-Signal.
    kspace_zf : torch.Tensor 
        Zero-filled, undersampled multi-coil k-space.
    R : int
        Undersampling factor.
    Nk_p : int
        GRAPPA-kernel size in PE-direction.
    Nk_r : int
        GRAPPA-kernel size in RO-direction.
        
    Returns
    -------
    prc_data : dict 
        Dictionary with keys: 'src': Source-matrix (ACS)
                                        'trg': Target-matrix (ACS)
                                        'src_pred' Source-matrix (zero-filled k-space)
    '''
    # number of coils, PE-lines and RO-lines in ACS and k-space
    (num_coil, num_p_acs, num_r_acs) = acs.shape

    # computing how many times GRAPPA kernel fits into the block
    # in read-direction (rep_r) and in phase-direction (rep_p)
    # for acs
    rep_r_acs = (num_r_acs - Nk_r) + 1
    rep_p_acs = num_p_acs - ((Nk_p - 1) * (R - 1) + Nk_p) + 1
    
    # total repetition-number (acs)
    rep_acs = rep_r_acs * rep_p_acs

    # dimension of source-vector
    dim_src_vec = Nk_r * Nk_p * num_coil

    # dimension of target-vector
    dim_trg_vec = (R) * num_coil

    # determining points within GRAPPA kernel which are related to
    # the target points
    r_trg = Nk_r // 2
    p_trg = Nk_p // 2 - 1

    # initializing the source - and target - matrices obtained from acs
    src_mat = torch.zeros((rep_acs, dim_src_vec), dtype=acs.dtype, requires_grad=False)
    trg_mat = torch.zeros((rep_acs, dim_trg_vec), dtype=acs.dtype, requires_grad=False)


    print('Loading Source - & Target Matrix ... ')
    start = time.time()
    # loop for kernel-displacement in phase-direction
    for p in range(rep_p_acs):
        # loop for kernel-displacement in read-direction
        for r in range(rep_r_acs):
            src_mat[p*rep_r_acs+r, ] = acs[:, p:p+(Nk_p-1)*R+1:R,
                                           r:r+(Nk_r-1) * 1 +
                                           1:1].reshape((dim_src_vec))
            trg_mat[p*rep_r_acs+r, ] = acs[:,
                                           p+p_trg*R+1:p+p_trg*R+R+1:1,
                                           r+r_trg].reshape((dim_trg_vec))

    print("Done" , "\n")
    end = time.time()
    print("Took ", '{:.3f}'.format(end - start), " sec \n")

    prc_data = {"src": src_mat, "trg": trg_mat}
    return prc_data


def extractSourceDatGrap(kspace_zf:torch.Tensor, R:int, Nk_p:int, Nk_r:int)->dict:    
    '''
    Function to get source- and target-matrix from acs to calculate
    GRAPPA kernel. Source-matrix from zero-filled k-space is also generated.
    
    Parameters
    ----------
    acs : torch.Tensor
        Auto-Calibration-Signal.
    kspace_zf : torch.Tensor
        Zero-filled, undersampled multi-coil k-space.
    R : int
        Undersampling factor.
    Nk_p : int
        GRAPPA-kernel size in PE-direction.
    Nk_r : int
        GRAPPA-kernel size in RO-direction.
        
    Returns
    -------
    prc_data : dict
        Dictionary with keys: 'src': Source-matrix (ACS)
                                        'trg': Target-matrix (ACS)
                                        'src_pred' Source-matrix (zero-filled k-space)
    '''
    # number of coils, PE-lines and RO-lines in ACS and k-space
    (num_coil, num_p_kspace, num_r_kspace) = kspace_zf.shape

    # for zero filled k-space
    rep_p_kspace = (num_p_kspace - ((Nk_p - 1) * (R - 1) + Nk_p)) // R + 1
    rep_r_kspace = num_r_kspace - Nk_r + 1
   
    # total repetition number (zero filled k-space)
    rep_kspace = rep_p_kspace * rep_r_kspace

    # dimension of source-vector
    dim_src_vec = Nk_r * Nk_p * num_coil

    # dimension of target-vector
    dim_trg_vec = (R) * num_coil

    # determining points within GRAPPA kernel which are related to
    # the target points
    r_trg = Nk_r // 2
    p_trg = Nk_p // 2 - 1

    # initializing a matrix that stores aquired
    # source-vectors from zero filled k-space as row-vectors
    src_pred_mat = torch.zeros((rep_kspace, num_coil * Nk_p * Nk_r),
                            dtype=kspace_zf.dtype, requires_grad=False)
                            

    # loop over repetitions in phase-direction
    for p in range(rep_p_kspace):
        # loop over repetitions in read-direction
        for r in range(rep_r_kspace):
            src_pred_mat[p*rep_r_kspace+r, ] = kspace_zf[:, p*R:p*R+(Nk_p-1) *
                                                  R+1:R, r:r+(Nk_r-1) +
                                                  1:1].reshape((num_coil *
                                                                Nk_p * Nk_r,
                                                                ))
    prc_data = {"src_pred": src_pred_mat}
    return prc_data

def fillDatGrap(kspace_zf:torch.Tensor, pred_mat:torch.Tensor, R:int, Nk_p:int, Nk_r:int)->torch.Tensor:
    '''
    Function to re-insert estimated missing singals back into k-space.
    
    Parameters
    ----------
    kspace_zf : torch.Tensor
        Zero-filled undersampled multi-coil k-space ([coils, PE, RO]).
    pred_mat : torch.Tensor
        Matrix containing predicted target-vectors.
    R : int
        Undersampling Factor.
    Nk_p : int
        GRAPPA Kernel in PE-direction.
    Nk_r : int GRAPPA Kernel in RO-direction.
    
    Returns
    -------
    kspace_zf : torch.Tensor
        GRAPPA interpolated k-space
    '''
    #recon_data = copy.deepcopy(data)
    # number of coils
    (num_coil, num_p_kspace, num_r_kspace) = kspace_zf.shape

    # for data-block
    rep_p_kspace = (num_p_kspace - ((Nk_p - 1) * (R - 1) + Nk_p))//R + 1
    rep_r_kspace = num_r_kspace - Nk_r + 1

    # determining the k-points within the kernel which are related to
    # the target points
    r_trg = Nk_r // 2
    p_trg = Nk_p // 2 - 1

    for p in range(rep_p_kspace):
        for r in range(rep_r_kspace):
            kspace_zf[:, p*R+p_trg*R+1:p*R+p_trg*R +
                    R+1:1, r+r_trg] = pred_mat[p*rep_r_kspace +
                                             r, ].reshape((num_coil, R, ))
    return kspace_zf


def weightMatCalc(src_mat:torch.Tensor, trg_mat:torch.Tensor, recon_param:float)->torch.Tensor:   
    '''
    This function computes the weight-matrix (w_mat) for given
    source- and target-matrices (src/trg_mat), in which the
    source-and target vectors are stored as row-vectors.
    The calculation is based on following steps:
    1. calculation of the pseudoinverse of src_mat with neglegation
    of singular-values that are below regularization parameter l
    2. multiplying the pseudoinverse of src_mat with target_matrix

    Parameters
    ----------
    src_mat : torch.Tensor 
        source-matrix, assuming following indexing: (
                repetition-index,
                source/target-vector
                )
    trg_mat : torch.Tensor
        target-matrix, same shape as source-matrix

    recon_param : float 
        Parameter below singular-values are neglegated in the pseudo-
            inverse-calculation.

    Returns
    -------
    w_mat : torch.tensor
        2-D torch tensor containing the weight-matrix (w_mat).
    '''
    # dimension of source-vector
    dim_src_vec = src_mat.shape[1]
    # dimension of target-vector
    dim_trg_vec = trg_mat.shape[1]
    # initializing weight-matrix
    w_mat = torch.zeros((dim_src_vec, dim_trg_vec),
                       dtype=src_mat.dtype, requires_grad=False)

    print('GRAPPA Weights Calculation...')
    start = time.time()
    w_mat[:, :] = torch.matmul(torch.linalg.pinv(src_mat[:, :], recon_param),
                            trg_mat[:, :])
                            
    print("Done", "\n")
    end = time.time()
    print("Took ", '{:.3f}'.format(end - start), "sec \n")
    return w_mat

def predTrgMat(src_pred_mat:torch.Tensor, w_mat:torch.Tensor)->torch.Tensor:
    '''
    Function to compute missing k-space signals.
    
    Parameters
    ----------
    src_pred_mat : torch.Tensor
        Source-Matrix from zero-filled k-space.
    w_mat : torch.Tensor
        GRAPPA Kernel.
    
    Returns
    -------
         Target-Matrix for zero filled k-space.
    '''    
    return torch.matmul(src_pred_mat, w_mat[:, :])


def trainGrappa(acs:torch.Tensor, R:int, kernel_design:dict, recon_param:float)->torch.Tensor:
    '''
    Function doing GRAPPA reconstruction. 
    
    Parameters
    ----------
    kspace_zf : torch.Tensor
        zero-filled k-space ([coils, PE, RO]).
    R : int
        Undersampling factor.
    kernel_design : dict
        Dictionary with keys 'phase' and 'read' for 
        GRAPPA kernel extension in PE- and RO-direction, respectively (e.g. kernel_design={'phase': 2, 'read':5}).
    recon_param : float Regularization parameter for SVD.
        
    Returns
    -------
    w_mat : torch.Tensor
        Reconstruction weights in k-space (a.k.a. GRAPPA kernel)
    '''    
    prc_data = extractTrainingDatGrap(acs=acs,
                              R=R,
                              Nk_p=kernel_design['phase'],
                              Nk_r=kernel_design['read'])


    w_mat = weightMatCalc(src_mat=prc_data["src"],
                          trg_mat=prc_data["trg"],
                          recon_param=recon_param)
    

    return w_mat


def inferGrappa(kspace_zf:torch.Tensor, w_mat:torch.Tensor, R:int, kernel_design:dict)->torch.Tensor:
    '''
    Function doing GRAPPA reconstruction. 
    
    Parameters
    ----------
    kspace_zf : torch.Tensor
        Zero-filled, undersampled multi-coil k-space ([coils, PE, RO]).
    w_mat : torch.Tensor 
        GRAPPA kernel
    R : torch.Tensor
        Undersampling factor.
    kernel_design : dict
        Dictionary with keys 'phase' and 'read' for 
    GRAPPA kernel extension in PE- and RO-direction, respectively (e.g. kernel_design={'phase': 2, 'read':5}).

        
    Returns
    -------
    grappa_reco : torch.Tensor
        Reconstructed multi-coil k-space in shape [coils, PE, RO].
    '''    
    prc_data = extractSourceDatGrap(
                              kspace_zf=kspace_zf,
                              R=R,
                              Nk_p=kernel_design['phase'],
                              Nk_r=kernel_design['read'])
    
    

    trg_pred_mat = predTrgMat(src_pred_mat=prc_data["src_pred"],
                              w_mat=w_mat)
    

    grappaReco = fillDatGrap(kspace_zf=kspace_zf,
                             pred_mat=trg_pred_mat,
                             R=R,
                             Nk_p=kernel_design['phase'],
                             Nk_r=kernel_design['read'])
    
    return grappaReco