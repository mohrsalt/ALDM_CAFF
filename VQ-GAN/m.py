#import torch

#ckpt = torch.load("/scratch/users/ntu/mohor001/stage2.ckpt", map_location="cpu")

# Keys:
#print(ckpt['state_dict'].keys())  # typically: ['state_dict', 'hyper_parameters', ...]

# Inspect embedding-related hyperparams
#print(ckpt["hyper_parameters"])

import torch

ckpt = torch.load("/scratch/users/ntu/mohor001/stage2.ckpt", map_location="cpu")
decoder_out_shape = ckpt["state_dict"]["decoder.conv_out.weight"].shape
print("Decoder output shape:", decoder_out_shape)
print("Output channels (modalities or image channels):", decoder_out_shape[0])

import torch

ckpt = torch.load("/scratch/users/ntu/mohor001/stage2.ckpt", map_location="cpu")
state_dict = ckpt["state_dict"]

disc_weight = state_dict["loss.discriminator.main.0.weight"]
print("Discriminator input channels:", disc_weight.shape[1])

