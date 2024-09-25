## train SBI using the flexible interface + generated data
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
import decimal
import time
import multiprocessing
from multiprocessing import Pool
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi.analysis import check_sbc, run_sbc, get_nltp, sbc_rank_plot

from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib import cm

from model_lightcurve import make_multiband_lc

import os 
import glob
import sys
import pickle
import io

#plt.style.use('~/jli184.mplstyle')

##
## set up file paths
##
dir_output = sys.argv[1]
dir_data = sys.argv[1]+'/data/'
n_train = 100000
study_name = 'LSTM_default_'+str(n_train)
model_name= dir_output+'{}_model.pt'.format(study_name)
pred_name= dir_output+'{}_pred.npy'.format(study_name)
embedding_net_name= dir_output+'{}_NN.pt'.format(study_name)
use_embedding_net = True


## choose CPU or GPU
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
print('Training on (CPU/GPU):',device)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

##
## read in train data + test data
##
print('Loading training/testing data.')
print('getting data from', dir_data)

lc_train = np.load(dir_data+'train_LC_100000_DRW_10x_SNR100.npy')
theta_train = np.load(dir_data+'train_params_100000_DRW_10x.npy')
theta_train = torch.tensor(theta_train.reshape(int(len(theta_train)/3),3),dtype=torch.float32)
lc_train = torch.tensor(lc_train.reshape(int(len(lc_train)/1260),1,1260),dtype=torch.float32)


lc_test = np.load(dir_data+'test_LC_1000_DRW_SNR100.npy')
theta_test = np.load(dir_data+'test_params_1000_DRW.npy')
theta_test = torch.tensor(theta_test.reshape(int(len(theta_test)/3),3),dtype=torch.float32)
lc_test = torch.tensor(lc_test.reshape(int(len(lc_test)/1260),1,1260),dtype=torch.float32)

print('finish loading training/testing data.')
print('training on ',theta_train.shape[0],' simulations')
print('testing on ',theta_test.shape[0],' simulations')

##
## set up embedding network (LSTM/CNN)
##
class ShallowLSTM(nn.Module):
    
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=14, num_layers=1):
        super(ShallowLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        
        x = x.view(-1,180,7)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # One time step (the last one)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        ## using only hidden states
        out = self.fc(out[:,-1,:])
        
        return out
        
class ShallowCNN2D(nn.Module):
    def __init__(self):
        super(ShallowCNN2D, self).__init__()
        # Reducing to a single Conv2D layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # Maxpool layer that reduces 32x32 image to 4x4
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=6 * 60 * 2, out_features=16)

    def forward(self, x):
        x = x.view(-1, 1, 180, 7)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6 * 60 * 2)
        x = F.relu(self.fc(x))
        return x

input_dim = 7  # Channels
hidden_dim = 64  # Number of LSTM cells
output_dim = 16  # Output features
num_layers = 1  # Single layer

if use_embedding_net is True: 
    embedding_net = ShallowLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    #embedding_net = ShallowCNN2D()
    print(summary(embedding_net,input_size=(1,1,1260)))
else:
    embedding_net = None

##
## train SBI, save the trained models
## or model exists, read from file
##
if os.path.exists(model_name) is False:

    prior = utils.BoxUniform(low=torch.tensor([np.cos(np.pi/4),5.,-2.]), high=torch.tensor([1.,10.,2.]),device=device)
    density_estimator_build = posterior_nn(model='maf', embedding_net=embedding_net, hidden_features=25, num_transforms=10)
    inference = SNPE(density_estimator=density_estimator_build, prior=prior,device=device)
    inference = inference.append_simulations(theta_train, lc_train)

    ## train the inference procedure
    start_time = time.time()
    density_estimator = inference.train(force_first_round_loss=True)
    posterior = inference.build_posterior(density_estimator)
    print('\nSBI training time',time.time()-start_time)
    ## save model
    torch.save(posterior, model_name)
    ## save embedding net
    with open(embedding_net_name, "wb") as handle:
        pickle.dump(density_estimator, handle)
    
else:
    print('Loading pre-trained model from file...')
    #model = torch.load(model_name)
    prior = utils.BoxUniform(low=torch.tensor([np.cos(np.pi/4),5.,-2.]), high=torch.tensor([1.,10.,2.]),device=device)
    ## save embedding net
    #with open(embedding_net_name, "rb") as handle:
    #    density_estimator = pickle.load(handle)
    
    if device == 'cpu':
        with open(embedding_net_name, "rb") as handle:
            density_estimator = CPU_Unpickler(handle).load()
            # = torch.load(io.BytesIO(handle), map_location=torch.device(device))
    else:
        with open(embedding_net_name, "rb") as handle:
            density_estimator = pickle.load(handle)
            
    density_estimator_build = posterior_nn(model='maf', embedding_net=embedding_net, hidden_features=25, num_transforms=10)

    inference = SNPE(density_estimator=density_estimator_build, prior=prior,device=device)
    posterior = inference.build_posterior(density_estimator)

##
## make prediction for all test case
##
theta_test = np.array(theta_test)
if os.path.exists(pred_name) is False:
    theta_pred = np.zeros((theta_test.shape[0],3))
    theta_low_pred = np.zeros((theta_test.shape[0],3))
    theta_high_pred = np.zeros((theta_test.shape[0],3))
    mmdot_pred = np.zeros((theta_test.shape[0],3))
    
    print('Start evaluation on the test data')
    start_time = time.time()
    for i in range(theta_test.shape[0]):
        
        posterior_samples = posterior.sample((10000,), x=torch.tensor(lc_test[i],device=device),show_progress_bars=False)
        posterior_samples = posterior_samples.cpu()
        posterior_samples[0,:] = np.cos(posterior_samples[0,:])
        theta_pred[i,:] = np.percentile(posterior_samples,50,axis=0)
        theta_low_pred[i,:] = np.percentile(posterior_samples,16,axis=0)
        theta_high_pred[i,:] = np.percentile(posterior_samples,84,axis=0)
        mmdot_pred[i,:] = np.percentile(posterior_samples[:,1]+posterior_samples[:,2],[16,50,84])
                
    print('\nTotal evaluation time',time.time()-start_time, (time.time()-start_time)/theta_test.shape[0]*1000,'/1000 it')
    np.save(pred_name,np.array([theta_pred.flatten(),theta_low_pred.flatten(),theta_high_pred.flatten(),mmdot_pred.flatten()]).T)
else:
    print('Reading prediction output from file...')
    theta_pred, theta_low_pred, theta_high_pred, mmdot_pred = np.load(pred_name).T
    theta_pred = theta_pred.reshape(theta_test.shape[0],3)
    theta_low_pred = theta_low_pred.reshape(theta_test.shape[0],3)
    theta_high_pred = theta_high_pred.reshape(theta_test.shape[0],3)
    mmdot_pred = mmdot_pred.reshape(theta_test.shape[0],3)
      
##
## plot prediction results
##

### pred vs true
labels = [r'cos($i$)',r'log$M_{\rm BH}$',r'log$\dot{M}$']
fig, ax = plt.subplots(1,3,figsize=(11,4))

for iax in range(3):
    
    ax[iax].errorbar(theta_test[:,iax],theta_pred[:,iax],fmt='ko',alpha=0.2,\
                     yerr=[theta_pred[:,iax]-theta_low_pred[:,iax],theta_high_pred[:,iax]-theta_pred[:,iax]])   
                     
    mae = np.sum(np.abs(theta_test[:,iax]-theta_pred[:,iax]))/len(theta_pred[:,iax])
    print(labels[iax], 'MAE=%.2f'%(mae), mae)
    
    ax[iax].plot([np.min(theta_test[:,iax]),np.max(theta_test[:,iax])],[np.min(theta_test[:,iax]),np.max(theta_test[:,iax])],'r-',zorder=100,alpha=0.5)
    ax[iax].set_ylim([np.min(theta_test[:,iax]),np.max(theta_test[:,iax])])
    ax[iax].set_xlim([np.min(theta_test[:,iax]),np.max(theta_test[:,iax])])
    ax[iax].text(np.min(theta_test[:,iax])+0.05*np.ptp(theta_test[:,iax]),np.min(theta_test[:,iax])+0.9*np.ptp(theta_test[:,iax]),'MAE=%.2f'%(mae),fontsize=14)
    ax[iax].set_xlabel('Truth',fontsize=14)
    ax[iax].set_ylabel('Prediction',fontsize=14)
    ax[iax].set_title(labels[iax],fontsize=16) 
    
plt.tight_layout()
plt.savefig(dir_output+'/plot/{}_pred_summary_mae.pdf'.format(study_name),bbox_inches='tight')
    
### d(lag)
fig, ax = plt.subplots(1,3,figsize=(11,3.5))

for iax in range(3):
    
    diff = theta_pred[:,iax]-theta_test[:,iax]
    hist, binedges = np.histogram(diff,bins=60,range=(-10.*np.std(diff),10*np.std(diff)))
    print(np.percentile(diff, [16,50,84]))
    
    ax[iax].hist(diff,range=(-10.*np.std(diff),10*np.std(diff)),bins=60,alpha=0.5,color='royalblue')
    ax[iax].set_xlim([-3.*np.std(diff),3*np.std(diff)])
    ax[iax].text(1.2*np.std(diff),np.max(hist)*0.9,r'$+$%.2f'%(np.percentile(diff,84)-np.percentile(diff,50)),fontsize=14)
    ax[iax].text(1.2*np.std(diff),np.max(hist)*0.8,r'$-$%.2f'%(np.percentile(diff,50)-np.percentile(diff,16)),fontsize=14)
    ax[iax].axvline(x=np.percentile(diff, 16),ls=':',c='k')
    ax[iax].axvline(x=np.percentile(diff, 84),ls=':',c='k')
    ax[iax].axvline(x=np.percentile(diff, 50),ls='-',c='k')
    ax[iax].set_xlabel('$\Delta$'+labels[iax],fontsize=16)
    ax[iax].set_ylabel('N',fontsize=16)
    
plt.tight_layout()
plt.savefig(dir_output+'/plot/{}_pred_hist.pdf'.format(study_name),bbox_inches='tight')


### d(lag)/err
fig, ax = plt.subplots(1,3,figsize=(11,3.5))

for iax in range(3):
    
    diff = (theta_pred[:,iax]-theta_test[:,iax])/((theta_high_pred[:,iax]-theta_low_pred[:,iax])/2.)  # dlag/err
    hist, binedges = np.histogram(diff,bins=30,range=(-3.,3.))
    mu, std = norm.fit(diff)
    x = np.linspace(-5,5,num=200)
    p = norm.pdf(x, mu, std)
    gaus = norm.pdf(x, 0., 1.)
    print(labels[iax],' dlag/err, mean = %.2f, std = %.2f' % (mu, std))

    ax[iax].plot(x, p/np.max(p)*np.max(hist), 'royalblue', linewidth=2,alpha=0.5)
    ax[iax].hist(diff,range=(-3.,3.),bins=30,alpha=0.5,color='royalblue')
    ax[iax].set_xlim([-3.,3.])
    ax[iax].text(1.5,np.max(hist)*0.9,r'%.2f'%(std),fontsize=14)
    ax[iax].axvline(x=np.percentile(diff, 16),ls=':',c='k')
    ax[iax].axvline(x=np.percentile(diff, 84),ls=':',c='k')
    ax[iax].axvline(x=np.percentile(diff, 50),ls='-',c='k')
    ax[iax].set_xlabel('$\Delta/\sigma$('+labels[iax]+')',fontsize=16)
    ax[iax].set_ylabel('N',fontsize=16)
    
plt.tight_layout()
plt.savefig(dir_output+'/plot/{}_err_hist.pdf'.format(study_name),bbox_inches='tight')

