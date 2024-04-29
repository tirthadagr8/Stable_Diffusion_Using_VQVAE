import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import glob
from sd.util import util
from PIL import Image
from torchvision import transforms as T

from sd.parameters import hp
# 'C:/Users/tarun/Downloads/Compressed/celeba_data/CelebA-HQ-img/*.jpg'



class CelebaDataset(Dataset):
    def __init__(self,im_path,annot_path,im_size,indices):
        self.im_path=im_path
        self.annot_path=annot_path
        self.indices=indices
        self.im_size=im_size
        
        self.images=glob.glob(im_path+'*.jpg')
        self.images+=glob.glob(im_path+'*.png')
        self.images+=glob.glob(im_path+'*.jpeg')
        
        for i in range(len(self.images)):
            self.images[i]=self.images[i].replace('\\','/')
        
        self.transforms=T.Compose([
                        T.Resize(self.im_size),
                        T.CenterCrop(self.im_size),
                        T.ToTensor(),
                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx=self.indices[idx]
        im_path=self.images[idx]
        name=im_path.split('/')[-1][:-4]
        captions_im = []
        with open(self.annot_path+name+'.txt') as f:
            for line in f.readlines():
                captions_im.append(line.strip())
        img=Image.open(im_path).convert('RGB').resize((self.im_size,self.im_size))
        
        return self.transforms(img),captions_im[0]
    

if __name__=='__main__':
    unique_id=np.arange(0,30000)
    threshold=int(unique_id.shape[0]*0.9)
    train_id=unique_id[:threshold]
    test_id=unique_id[threshold:]
   
    train_dl=DataLoader(CelebaDataset(hp.im_path,hp.annot_path,hp.im_size,train_id),batch_size=hp.batch_size)
    val_dl=DataLoader(CelebaDataset(hp.im_path,hp.annot_path,hp.im_size,test_id),batch_size=hp.batch_size) 
    
