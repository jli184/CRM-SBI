import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import multiprocessing
from multiprocessing import Pool
import time
import os
from functools import partial

from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi
from sbi.utils.get_nn_models import posterior_nn

from model_lightcurve import make_multiband_lc

dir_output = sys.argv[1]
nthreads = int(os.environ.get('SLURM_CPUS_ON_NODE', multiprocessing.cpu_count()))

print('running on # of cores: ',nthreads)

## set up prior range
## three parameters: cos(inc) (cos(pi/4), 1), logMbh (5-10), logMdot (-2,2)
num_dim = 3
prior = utils.BoxUniform(low=torch.tensor([np.cos(np.pi/4.),5.,-2.]), high=torch.tensor([1.,10.,2.]))


##
## making testing data
##
nsim=1000
thetas = prior.sample((int(nsim),))

xs_true = []
xs_noise = []
with multiprocessing.Pool(processes=nthreads) as pool:
	start_time = time.time()
	data_true, data_noise = zip(*pool.map(make_multiband_lc, thetas))
	print("Simulation time", time.time() - start_time, 'on',  nthreads, 'threads') 

xs_true.append(data_true)
xs_noise.append(data_noise)
xs_true = torch.as_tensor(xs_true[0],dtype=torch.float32)
xs_noise = torch.as_tensor(xs_noise[0],dtype=torch.float32)

np.save(dir_output+'test_params_'+str(int(nsim))+'_DRW.npy',thetas.flatten())
np.save(dir_output+'test_LC_'+str(int(nsim))+'_DRW_true.npy',xs_true.flatten())
np.save(dir_output+'test_LC_'+str(int(nsim))+'_DRW_SNR100.npy',xs_noise.flatten())


##
## make training data
##
nsim=100000
nrepeat = 10
thetas = prior.sample((int(nsim/nrepeat),))
thetas = thetas.repeat(nrepeat,1) ## make 10 iterations of the same prior in training data

xs_true = []
xs_noise = []
with multiprocessing.Pool(processes=nthreads) as pool:
	start_time = time.time()
	data_true, data_noise = zip(*pool.map(make_multiband_lc, thetas))
	print("Simulation time", time.time() - start_time, 'on',  nthreads, 'threads') 

xs_true.append(data_true)
xs_noise.append(data_noise)
xs_true = torch.as_tensor(xs_true[0],dtype=torch.float32)
xs_noise = torch.as_tensor(xs_noise[0],dtype=torch.float32)

np.save(dir_output+'train_params_'+str(int(nsim))+'_DRW_10x.npy',thetas.flatten())
np.save(dir_output+'train_LC_'+str(int(nsim))+'_DRW_10x_true.npy',xs_true.flatten())
np.save(dir_output+'train_LC_'+str(int(nsim))+'_DRW_10x_SNR100.npy',xs_noise.flatten())
