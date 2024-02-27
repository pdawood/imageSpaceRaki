
import torch
import numpy as np 
import matplotlib.patches as patches
import time

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


from cnn.utils_imgSpc.fft import IFFT2, FFT2
from cnn.rakiModels import trainRaki, inferRaki
from grappa.grappaReco import trainGrappa, inferGrappa
from grappa.grappaRecoImg import kspc2ImgWeights, grappaImg

from utils.metrics import getMetrics

from cnn.utils_imgSpc.act2toelpitz import rakiImgRecoToel
from cnn.rakiModels_imgSpc import getImgSpcParams, rakiImgSpc

from noiseEval.analyticalFunc import rakiGfactorCalc, grappaGfactorCalc
from noiseEval.prmFunc import prmToel, prmGrappaImg
from noiseEval.autodiffFunc import autoDiffToel, autoDiffGrappa

# This code assumes multi-channel, pre-whitened 2D k-space in shape (coils, PE, RO), with PE: Phase-Encoding direction and RO: Read-Out direction.
# First, the fully-sampled k-space and the undersampling mask is loaded. 

# chose  dataset
head = 'flash' # 'flash' or 'tse'
# chose acceleration factor
R = 4  # 4-5 for flash, 5-6 for tse
# The number of ACS lines 
NACS = 40


if head == 'flash':
    pathToPreWhitenddata = './data/neuro_flash/'
elif head== 'tse':
    pathToPreWhitenddata = './data/neuro_tse/'
    
kspace = np.load(pathToPreWhitenddata+'kspace.npy')
mask = np.load(pathToPreWhitenddata+'mask.npy')

# check k-space scaling such that minimum signal has order of magnitude 0     
kspace_fs = np.copy(kspace)
scaling = np.floor(np.log10(np.min(np.abs(kspace_fs[np.where(kspace_fs!=0)]))))
sclFac = 10**(-1*int(scaling))
kspace_fs *= sclFac
print(f'Scaling factor {sclFac}')

nC, nP, nR = kspace.shape
nPorig, nRorig = nP, nR
ipat = np.zeros((nP, nR))
ipat[::R,:] = 1 
ipat[int((nP-NACS)/2):int((nP+NACS)/2),:] = 1
acq = np.where(ipat[:,0]==1)[0] # get index of first non-zero sample
kspace_fs = kspace_fs[:,acq[0]:,:] # the code does not allow for leading zeros in k-space 
ipat = ipat[acq[0]:,:]
(nC, nP, nR) = kspace_fs.shape # (coils, number phase encoding lines, number read out lines)
R = acq[1] - acq[0] # acceleration factor 
kspace_zf = np.zeros_like(kspace_fs) # generate zero-filled kspace without acs  
kspace_zf[:,::R,:] = kspace_fs[:,::R,:]

# Lets get the ACS from the center of the fully sampled k-space. For 4-fold undersampling, we use 18 ACS lines.

acq_conti = np.where(acq[1:]-acq[:-1]==1)[0] 
acs_start = acq[acq_conti[0]]
acs_end = acq[acq_conti[-1]]+1
acs = kspace_fs[:,acs_start:acs_end+1,:]
(_,nP_acs, nR_acs) = acs.shape 

R_eff = ((((nP-nP_acs)/R)+nP_acs)/nP)**-1
print('Acceleration Factor: ', str(R), '\n', 'Number ACS Lines: ', str(nP_acs), '/', str(nP), '\n', 'Effective Acceleration Factor: ', '{:.2f}'.format(R_eff))

# Now we build the RAKI-Network. The dictionary 'layer_design_raki' specifies its architechture.  
layer_design_raki = {'num_hid_layer': 2, # number of hidden layers, in this case, its 2
                    'input_unit': nC,    # number channels in input layer, nC is coil number 
                        1:[128,(2,5)],   # the first hidden layer has 256 channels, and a kernel size of (2,5) in PE- and RO-direction
                        2:[64,(1,1)],   # the second hidden layer has 128 channels, and a kernel size of (1,1) in PE- and RO-direction
                    'output_unit':[(R)*nC,(1,5)] # the output layer has (R-1)*nC channels, and a kernel size of (1,5) in PE- and RO-direction
                    }

kspace_zf = torch.from_numpy(kspace_zf)
kspace_fs = torch.from_numpy(kspace_fs)
sigma = torch.diag(torch.ones(nC,dtype=kspace_fs.dtype))

img_zf = torch.sum(torch.abs(IFFT2(kspace_zf))**2,axis=0)**0.5

net = trainRaki(torch.from_numpy(acs), R, layer_design_raki)
raki_reco = inferRaki(torch.clone(kspace_zf), R, layer_design_raki, net)
raki_reco = raki_reco.detach()

(imgW, actW) = getImgSpcParams(torch.clone(kspace_zf), net, R)
raki_reco_img = rakiImgSpc(IFFT2(torch.clone(kspace_zf)),imgW, actW)
raki_reco_img = raki_reco_img.detach()
raki_reco_img *= 1/np.sqrt(nP*nR)

kernel_grappa = {'phase':2, 'read':5}
w_mat_grappa = trainGrappa(torch.from_numpy(acs),  R, kernel_grappa, 0.0001)
grappa_reco = inferGrappa(torch.clone(kspace_zf), w_mat_grappa, R, kernel_grappa)
grappa_reco = grappa_reco.detach()

w_im_grappa = kspc2ImgWeights(w_mat_grappa, kernel_grappa, R, [nC, nP, nR])
grappa_reco_img = grappaImg(IFFT2(torch.clone(kspace_zf)), w_im_grappa)
grappa_reco_img = grappa_reco_img.detach()

reference = IFFT2(kspace_fs)
raki_recoFFT = IFFT2(raki_reco)
grappa_recoFFT = IFFT2(grappa_reco)

reference = torch.sum(torch.abs(reference)**2,axis=0)**0.5
raki_recoFFT = torch.sum(torch.abs(raki_recoFFT)**2,axis=0)**0.5  
raki_reco_imgSOS = torch.sum(torch.abs(raki_reco_img)**2,axis=0)**0.5  
#raki_reco_imgSOS *= 1/np.sqrt(nP*nR)
grappa_recoFFT = torch.sum(torch.abs(grappa_recoFFT)**2,axis=0)**0.5  
grappa_reco_imgSOS = torch.sum(torch.abs(grappa_reco_img)**2,axis=0)**0.5  

diff_reference = torch.abs(reference - reference)
diff_raki = torch.abs(reference - raki_recoFFT)
diff_raki_imgSOS = torch.abs(reference - raki_reco_imgSOS)
diff_grappa = torch.abs(reference - grappa_recoFFT)
diff_grappa_imgSOS = torch.abs(reference - grappa_reco_imgSOS)

nmse_raki, ssim_raki, psnr_raki = getMetrics(reference, raki_recoFFT, mask)
nmse_raki_imgSOS, ssim_raki_imgSOS, psnr_raki_imgSOS = getMetrics(reference, raki_reco_imgSOS, mask)
nmse_grappa, ssim_grappa, psnr_grappa = getMetrics(reference, grappa_recoFFT, mask)
nmse_grappa_img, ssim_grappa_img, psnr_grappa_img = getMetrics(reference,grappa_reco_imgSOS, mask)

max_rss = torch.max(torch.abs(reference))
alpha = 0.6
diffScaling = 1e1

font = {'fontsize':24, 'color': 'white', 'fontweight':'bold'}
fig, axs = plt.subplots(2,5,figsize=[15,6], facecolor='k', dpi=300)
if head == 'flash':
    plt.text(-850,-500,'RAKI', fontdict=font)
    plt.text(-100,-500,'GRAPPA', fontdict=font)   
    plt.text(-1500,-500,'Reference', fontdict=font)
elif head == 'tse':
    plt.text(-700,-400,'RAKI', fontdict=font)
    plt.text(-100,-400,'GRAPPA', fontdict=font)
    plt.text(-1250,-400,'Reference', fontdict=font)
font = {'fontsize':22, 'color': 'white', 'fontweight': 'bold'}
axs[0,0].imshow(torch.abs(torch.rot90(reference*mask)), cmap='gray', vmin=0, vmax=max_rss*alpha)
axs[0,1].imshow(torch.abs(torch.rot90(raki_reco_imgSOS*mask)), cmap='gray', vmin=0, vmax=max_rss*alpha)
axs[0,2].imshow(torch.abs(torch.rot90(raki_recoFFT*mask)), cmap='gray', vmin=0, vmax=max_rss*alpha)
axs[0,3].imshow(torch.abs(torch.rot90(grappa_reco_imgSOS*mask)), cmap='gray', vmin=0, vmax=max_rss*alpha)
axs[0,4].imshow(torch.abs(torch.rot90(grappa_recoFFT*mask)), cmap='gray', vmin=0, vmax=max_rss*alpha)
axs[1,0].imshow(torch.abs(torch.rot90(diff_reference*mask*diffScaling)), cmap='gray', vmin=0, vmax=max_rss*alpha)
axs[1,1].imshow(torch.abs(torch.rot90(diff_raki_imgSOS*mask*diffScaling)), cmap='gray', vmin=0, vmax=max_rss*alpha)
axs[1,2].imshow(torch.abs(torch.rot90(diff_raki*mask*diffScaling)), cmap='gray', vmin=0, vmax=max_rss*alpha)
axs[1,3].imshow(torch.abs(torch.rot90(diff_grappa_imgSOS*mask*diffScaling)), cmap='gray', vmin=0, vmax=max_rss*alpha)
axs[1,4].imshow(torch.abs(torch.rot90(diff_grappa*mask*diffScaling)), cmap='gray', vmin=0, vmax=max_rss*alpha)
axs[0,1].set_title('Image space', fontdict=font)
axs[0,2].set_title('k-space', fontdict=font)
axs[0,3].set_title('Image space', fontdict=font)
axs[0,4].set_title('k-space', fontdict=font)
axs[0,0].text(-10,250, f'R = {R}', rotation=90, fontdict=font)
axs[1,0].text(-5,300, 'Error (x10)', rotation=90, fontdict=font)
font = {'fontsize':16, 'color': 'white'}
axs[1,1].text(0,50, ' NMSE \n SSIM \n PSNR', fontdict=font)
axs[1,1].text(200,50, f'{1e4*nmse_raki_imgSOS:.2f} \n {1e2*ssim_raki_imgSOS:.2f} \n {psnr_raki_imgSOS:.2f}', fontdict=font)
axs[1,2].text(200,50, f'{1e4*nmse_raki:.2f} \n {1e2*ssim_raki:.2f} \n {psnr_raki:.2f}', fontdict=font)
axs[1,3].text(200,50, f'{1e4*nmse_grappa_img:.2f} \n {1e2*ssim_grappa_img:.2f} \n {psnr_grappa_img:.2f}', fontdict=font)
axs[1,4].text(200,50, f'{1e4*nmse_grappa:.2f} \n {1e2*ssim_grappa:.2f} \n {psnr_grappa:.2f}', fontdict=font)
if head == 'flash':
    x1, x2, y1, y2 = 100,150,110,160    
elif head == 'tse':
    x1, x2, y1, y2 = 105,155,110,160   
axins = zoomed_inset_axes(axs[0,0], 2, loc=4)
axins.imshow(torch.abs(torch.rot90(torch.flip(reference, (1,)))), cmap='gray', vmin=0, vmax=max_rss*alpha)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_yticks([])
axins.set_xticks([])
for spine in axins.spines.values():
    spine.set_edgecolor('white')
if head=='flash':
    rect = patches.Rectangle((x1, y2), x2-x1, y2-y1, linewidth=1, edgecolor='white', facecolor='none')
if head=='tse':
    rect = patches.Rectangle((x1, y2-65), x2-x1, y2-y1, linewidth=1, edgecolor='white', facecolor='none')
axs[0,0].add_patch(rect)
axins = zoomed_inset_axes(axs[0,1], 2, loc=4)
axins.imshow(torch.abs(torch.rot90(torch.flip(raki_reco_imgSOS, (1,)))), cmap='gray', vmin=0, vmax=max_rss*alpha)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_yticks([])
axins.set_xticks([])
for spine in axins.spines.values():
    spine.set_edgecolor('white')
axins = zoomed_inset_axes(axs[0,2], 2, loc=4)
axins.imshow(torch.abs(torch.rot90(torch.flip(raki_recoFFT, (1,)))), cmap='gray', vmin=0, vmax=max_rss*alpha)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_yticks([])
axins.set_xticks([])
for spine in axins.spines.values():
    spine.set_edgecolor('white')
axins = zoomed_inset_axes(axs[0,3], 2, loc=4)
axins.imshow(torch.abs(torch.rot90(torch.flip(grappa_reco_imgSOS, (1,)))), cmap='gray', vmin=0, vmax=max_rss*alpha)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_yticks([])
axins.set_xticks([])
for spine in axins.spines.values():
    spine.set_edgecolor('white')
axins = zoomed_inset_axes(axs[0,4], 2, loc=4)
axins.imshow(torch.abs(torch.rot90(torch.flip(grappa_recoFFT, (1,)))), cmap='gray', vmin=0, vmax=max_rss*alpha)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_yticks([])
axins.set_xticks([])
for spine in axins.spines.values():
    spine.set_edgecolor('white')
plt.show()

del imgW, actW


cropN = 12
cP = int((nP-cropN)//2)-(int((nP-cropN)//2)%R)
cR = int((nR-cropN)//2)

kspace_zf_crop = kspace_zf[:,cP:cP+cropN,cR:cR+cropN]
kspace_fs_crop = kspace_fs[:,cP:cP+cropN,cR:cR+cropN]
_, nP, nR = kspace_zf_crop.shape

start = time.time()
(imgWcrop, actWcrop) = getImgSpcParams(kspace_zf_crop, net, R)

(test, toelW) = rakiImgRecoToel(IFFT2(kspace_zf_crop), imgWcrop, actWcrop)
img_rss = torch.sum(torch.abs(test)**2,axis=0)**0.5
p_raki = (torch.conj(test[:,:,:])/img_rss[None, :,:]).reshape(nC, nP*nR)
gFacRaki = rakiGfactorCalc(imgWcrop, actWcrop, p_raki, R, sigma)
end = time.time()
print(f'Took RAKI gfactor analytical: {end-start:.2f} secs')

start = time.time()
w_im_grappa = kspc2ImgWeights(w_mat_grappa, kernel_grappa, R, [nC, nP, nR])
grappa_reco_img_C = grappaImg(IFFT2(torch.clone(kspace_zf_crop)), w_im_grappa)
grappa_reco_img_C = grappa_reco_img_C.detach()
grappa_reco_img_C_rss = torch.sum(torch.abs(grappa_reco_img_C)**2,axis=0)**0.5
p_grappa = (torch.conj(grappa_reco_img_C[:,:,:])/grappa_reco_img_C_rss[None, :,:]).reshape(nC, nP*nR)
gFacGrappa = grappaGfactorCalc(w_im_grappa, p_grappa, R, [nC, nP, nR],sigma)
end = time.time()
print(f'Took GRAPPA gfactor analytical {end-start:.2f} secs')


print('done')

start = time.time()
gFacRakiAD = autoDiffToel(kspace_zf_crop, imgWcrop, actWcrop, toelW, p_raki, R, sigma)
end = time.time()
print(f'Took RAKI gfactor autodiff {end-start:.2f} secs')

start = time.time()
gFacGrappaAD = autoDiffGrappa(kspace_zf_crop, w_im_grappa, p_grappa, R, sigma)
end = time.time()
print(f'Took Grappa gfactor autodiff {end-start:.2f} secs')

start = time.time()
gFacRakiPRMT = prmToel(kspace_zf_crop, imgWcrop, actWcrop, sclFac, kspace_fs_crop, R, toelW, sigma)
end = time.time()
print(f'Took RAKI gfactor MC {end-start:.2f} secs')

start = time.time()
gFacGrappaImgPRM = prmGrappaImg(kspace_zf_crop, w_im_grappa, sclFac, kspace_fs_crop, R, sigma)
end = time.time()
print(f'Took GRAPPA gfactor MC {end-start:.2f} secs')

maxGplot = {}
maxGplot[4] = 3
maxGplot[5] = 5
maxGplot[6] = 5

maxG = maxGplot[R]

font = {'fontsize':36, 'color': 'k', 'fontweight': 'bold'}
fig, axs = plt.subplots(2,3, figsize=[12,8], dpi=100, constrained_layout=True)
axs[0,0].imshow(torch.rot90(gFacRaki).detach(), vmin=0, vmax=maxG)
axs[1,0].imshow(torch.rot90(gFacGrappa).detach(), vmin=0, vmax=maxG)
axs[0,1].imshow(torch.rot90(gFacRakiPRMT).detach(), vmin=0, vmax=maxG)
axs[1,1].imshow(torch.rot90(gFacGrappaImgPRM).detach(), vmin=0, vmax=maxG)
pcm0 = axs[0,2].imshow(torch.rot90(gFacRakiAD).detach(), vmin=0, vmax=maxG)
pcm1 = axs[1,2].imshow(torch.rot90(gFacGrappaAD).detach(), vmin=0, vmax=maxG)
for jj in range(2):
    for kk in range(3):
        axs[jj, kk].axis('off')
axs[0,0].set_title('Analytical', fontdict=font)
axs[0,1].set_title('Monte Carlo', fontdict=font)
axs[0,2].set_title('Autodiff', fontdict=font)
axs[0,0].text(-10,30, 'RAKI', rotation=90, fontdict=font)
axs[1,0].text(-10,40, 'GRAPPA', rotation=90, fontdict=font)
cb0 = fig.colorbar(pcm0, ax=axs[0,:], shrink=1, ticks=[0,1,2,3,4,5])
cb1 = fig.colorbar(pcm1, ax=axs[1,:], shrink=1, ticks=[0,1,2,3,4,5])
cb0.ax.tick_params(labelsize=36)
cb1.ax.tick_params(labelsize=36)
plt.show()

print('done')

import matplotlib
mask_crop = IFFT2(FFT2(torch.from_numpy(mask))[cP:cP+cropN,cR:cR+cropN]).detach().numpy().real
mask_crop = np.where(mask_crop>1,int(1),int(0))
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 62}
matplotlib.rc('font', **font)
fig, axs = plt.subplots(1,2, figsize=[28,18], dpi=200)
xlim = [0,maxG]
ylim = [0,maxG]
axs[0].scatter((gFacRaki*mask_crop).flatten(), (gFacRakiPRMT*mask_crop).flatten())
axs[1].scatter((gFacGrappa*mask_crop).flatten(),(gFacGrappaImgPRM*mask_crop).flatten())
axs[0].plot(xlim, ylim, color='k')
axs[1].plot(xlim, ylim, color='k')
for jj in range(2):
    axs[jj].set_xlim(xlim)
    axs[jj].set_ylim(ylim)
axs[0].grid('one')
axs[1].grid('one')
fig.text(0.5, 0.04, 'Analytical', ha='center')
fig.text(0.04, 0.5, 'Monte Carlo', va='center', rotation='vertical')
axs[0].set_title('RAKI', fontdict=font)
axs[1].set_title('GRAPPA', fontdict=font)

axs[0].set_yticks(np.arange(0,maxG+1))
axs[1].set_yticks(np.arange(0,maxG+1))
axs[0].set_xticks(np.arange(1,maxG+1))
axs[1].set_xticks(np.arange(1,maxG+1))

plt.show()

print('done')
