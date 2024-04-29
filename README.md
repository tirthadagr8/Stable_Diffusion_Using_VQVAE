# Stable_Diffusion_Using_VQVAE

I first trained the VQVAE(Vector Quantized Variational Auto Encoder) using the celeba dataset containing 200000+ images, it can be found in [kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)<br>

After 2 Epochs, this is a sample from VQVAE:<br>
![](https://github.com/tirthadagr8/Stable_Diffusion_Using_VQVAE/blob/main/old_files/vqvae_sample.png?raw=true)<br>
Model weights can be found [here](https://www.kaggle.com/models/mastersincsgo/vqvae_ckpt/PyTorch/vqvae_ckpt_celeba/1).<br>

The dataset used for training the UNET model is [CelebaText](https://www.kaggle.com/datasets/mastersincsgo/celebatext)<br>
Then using the trained VQVAE, we trained the diffusion model. Each epoch was taking bit longer, so I decided to stop is at 5 epochs, and below is a sample from it after giving it a prompt-'He is a man'<br>
![](https://github.com/tirthadagr8/Stable_Diffusion_Using_VQVAE/blob/main/old_files/epoch4.png?raw=true)<br>
