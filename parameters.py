class Params:
    im_path='C:/Users/tarun/Downloads/Compressed/celeba_data/CelebA-HQ-img/'
    annot_path='C:/Users/tarun/Downloads/Compressed/celeba_data/celeba-caption/'
    vqvae_ckpt='C:/Users/tarun/Downloads/Compressed/ver1.pth'
    im_size=256
    temb_dim=512
    batch_size=4
    text_model='bert'
    lr=5e-06
    num_epochs=5
    num_timesteps=1000
    beta_start=0.00085
    beta_end=0.012
    cf_guidance_scale=0.75
    # UNET
    in_channels=3
    conv_out_channels=128
    norm_channels=32
    num_heads=8
    text_embed_dim=768
    down_blocks=[256, 384, 512, 768]
    mid_blocks=[768,512]
    num_layers=2

hp=Params()
