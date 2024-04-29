import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import glob
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset

from sd.parameters import hp
from sd.diffusion import Diffusion
from sd.VQVAE import VQVAE
from sd.util import util
from sd.dataset import CelebaDataset
from sd.ddpm import LinearNoiseScheduler
from sd.pipeline import train,inference
from sd.UNET import UNET


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Load Dataset
unique_id=np.arange(0,30000)
threshold=int(unique_id.shape[0]*0.9)
train_id=unique_id[:threshold]
test_id=unique_id[threshold:]

train_dl=DataLoader(CelebaDataset(hp.im_path,hp.annot_path,hp.im_size,train_id),batch_size=hp.batch_size)
val_dl=DataLoader(CelebaDataset(hp.im_path,hp.annot_path,hp.im_size,test_id),batch_size=hp.batch_size)

# Load Models
vqvae=VQVAE()
diffusion=UNET()
assert os.path.exists(hp.vqvae_ckpt),"Model weights Needed to Proceed"
if  os.path.exists(hp.vqvae_ckpt):
    vqvae.load_state_dict(torch.load(hp.vqvae_ckpt,map_location=device))
    print('Model Weights Loaded')
else:
    print('Model weights Not Loaded')

# Load helper objects
scheduler=LinearNoiseScheduler(num_timesteps=hp.num_timesteps,
                               beta_start=hp.beta_start,beta_end=hp.beta_end)

optimizer=optim.Adam(diffusion.parameters(),lr=hp.lr)

train(diffusion,vqvae,train_dl,optimizer,scheduler,device)
inference(diffusion,vqvae,scheduler,device)

