import torch

ckpt = torch.load("/home/users/ntu/mohor001/BraSyn_2025_Task8/VQ-GAN/logs/2025-06-25T14-32-47_brats_vqgan_stage1/lightning_logs/2025-06-25T14-32-47_brats_vqgan_stage1/checkpoints/epoch=1-step=1492.ckpt", map_location="cpu")

# Keys:
print(ckpt['state_dict'].keys())  # typically: ['state_dict', 'hyper_parameters', ...]

# Inspect embedding-related hyperparams
#print(ckpt["hyper_parameters"])

import torch

#ckpt = torch.load("/home/users/ntu/mohor001/BraSyn_2025_Task8/VQ-GAN/logs/2025-06-25T03-14-28_brats_vqgan_stage1/lightning_logs/2025-06-25T03-14-28_brats_vqgan_stage1/checkpoints/epoch=1-step=1492.ckpt", map_location="cpu")
#decoder_out_shape = ckpt["state_dict"]["decoder.conv_out.weight"].shape
#print("Decoder output shape:", decoder_out_shape)
#print("Output channels (modalities or image channels):", decoder_out_shape[0])

import torch

#ckpt = torch.load("/home/users/ntu/mohor001/BraSyn_2025_Task8/VQ-GAN/logs/2025-06-25T03-14-28_brats_vqgan_stage1/lightning_logs/2025-06-25T03-14-28_brats_vqgan_stage1/checkpoints/epoch=1-step=1492.ckpt", map_location="cpu")
#state_dict = ckpt["state_dict"]

#disc_weight = state_dict["loss.discriminator.main.0.weight"]
#print("Discriminator input channels:", disc_weight.shape[1])

