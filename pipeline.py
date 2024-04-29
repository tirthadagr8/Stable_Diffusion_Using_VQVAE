import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from sd.parameters import hp
from sd.util import util


def inference(diffusion,vqvae,scheduler,device):
    diffusion.eval().to(device)
    vqvae.eval().to(device)
    with torch.no_grad():
        xt = torch.randn((1,
                      hp.in_channels,
                      hp.im_size//8,
                      hp.im_size//8)).to(device)
        text_prompt = ['He is a man.']
        text_prompt_embed = util.get_text_representation(text_prompt).to(device)
        cf_guidance_scale = hp.cf_guidance_scale
        for i in tqdm(reversed(range(hp.num_timesteps))):
            t = (torch.ones((xt.shape[0],)) * i).long().to(device)
            t=util.get_time_embedding(t,hp.temb_dim).to(device)
            noise_pred_cond = diffusion(xt, t, text_prompt_embed)
            if cf_guidance_scale > 1:
                noise_pred_uncond = diffusion(xt, t, text_prompt_embed)
                noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
            if i == 0:
                # Decode ONLY the final iamge to save time
                ims = vqvae.decode(xt)
                img=torchvision.transforms.ToPILImage()(ims[0])
                plt.imshow(img)
                plt.show()
                img.save('sample1.png')
            else:
                ims = x0_pred
        

def train(diffusion,vqvae,dataloader,optimizer_diffusion,scheduler,device):
    criterion=nn.MSELoss()
    diffusion.train().to(device)
    vqvae.eval().to(device)
    losses=[]
    for epoch in range(hp.num_epochs):
        for im,txt in tqdm(dataloader):
            txt=util.get_text_representation(txt)
            im,txt=im.to(device),txt.to(device)
            optimizer_diffusion.zero_grad()
            im,_=vqvae.encode(im)
            #print(im.shape)
            noise = torch.randn_like(im).to(device)
            t = torch.randint(0, hp.num_timesteps, (im.shape[0],)).to(device)
            noisy_im = scheduler.add_noise(im, noise, t)
            t=util.get_time_embedding(t,hp.temb_dim).to(device)
            noise_pred = diffusion(noisy_im, t, txt)
            loss = criterion(noise_pred, noise.detach())
            losses.append(loss.item())
            loss.backward()
            noise_pred=noise_pred.detach().cpu()
            loss=loss.detach().cpu()
            optimizer_diffusion.step()
            
        print('Finished epoch:{} | Loss : {:.4f}'.format(epoch + 1,np.mean(losses)))
        torch.save(diffusion.state_dict(), 'diffusion.pth')
        inference(diffusion,vqvae,scheduler,device)

    

