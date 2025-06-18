import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import copy
from main import instantiate_from_config

from taming.models.normalization import SPADEResnetBlock, SPADEGenerator
from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer

class CHattnblock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, dim, 1),
            nn.SiLU(),
            nn.Conv3d(dim, dim, 1),
            nn.Sigmoid())

    def forward(self, x):
        w = self.attn(x)
        # print(w.shape)
        return w
    
class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 modalities=[], # list of modalities to use
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 stage=1,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        ddconfig_new = copy.deepcopy(ddconfig)
        ddconfig_new['in_channels'] = 3
        self.encoder_complementary = Encoder(**ddconfig_new) #new
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape) 
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)

        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
        self.attn_blocks = nn.ModuleList([CHattnblock(128*2) for i in range(5)]) #new
        self.conv1 = nn.Conv3d(512, 256, 1) #new
        
        self.modalities = modalities
        self.spade = SPADEGenerator(modalities)
        self.conv_out_enc = torch.nn.Conv3d(256,
                                         3,
                                         kernel_size=3,
                                        stride=1,
                                         padding=1) #new
        self.stage = stage
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        return h
        
    def encode_comp(self, x):
        h = self.encoder_complementary(x)
        return h
    
    def quantizer(self, h):
        h = self.conv_out_enc(h)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)

        return quant, emb_loss, info
    
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, target=None, input_modals=None):
        if target is None:
            h=self.encode(input)
        else:
            h1=self.encode(input[:,0])
            h2=self.encode(input[:,1])
            h3=self.encode(input[:,2])
            h_comp=self.encoder_complementary(input)
            h_concat=torch.concat([h1,h2,h3,h_comp],dim=1)
            h=self.caff(h_concat,input_modals)
        #caff
        quant, diff, _ = self.quantizer(h)
        if target is not None: quant = self.spade(quant, target)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        
        return x.float()
    
    def caff(self, net_z, chosen_sources): #check functional similarity to hfeconder
        """
        net_z: Tensor of shape (B, 4, C, H, W)
        chosen_sources: list of 3 indices from [0, 1, 2, 3] indicating input modalities
        """
        comp_features = net_z[:, -1] 
        x_fusion_s = torch.zeros_like(comp_features)
        x_fusion_h = torch.zeros_like(comp_features)

        raw_attns = []  
        for i, src in enumerate(chosen_sources):
            attn_map = self.attn_blocks[src](net_z[:, i]) 
            raw_attns.append(attn_map.unsqueeze(1))  

    
        comp_attn = self.attn_blocks[-1](net_z[:, -1]).unsqueeze(1)  
        raw_attns.append(comp_attn)

        x_attns = torch.cat(raw_attns, dim=1)  

    
        for i in range(3):
            x_fusion_s += net_z[:, i] * x_attns[:, i]
        x_fusion_s += net_z[:, -1] * x_attns[:, -1]

    
        x_attns_soft = F.softmax(x_attns[:, :3], dim=1)  
        x_attns = torch.cat([x_attns_soft, x_attns[:, 3:]], dim=1)

        for i in range(3):
            x_fusion_h += net_z[:, i] * x_attns[:, i]
        x_fusion_h += net_z[:, -1]  # raw residual (no attention weight)

        
        x_fusion = self.conv1(torch.cat((x_fusion_s, x_fusion_h), dim=1))  
        return x_fusion
    
    def modalities_to_indices(self,source):
        return [self.modalities.index(mod) for mod in source]

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        target = random.choice(self.modalities)
        source = [m for m in self.modalities if m != target]
        src_idx=self.modalities_to_indices(source)
        x_tar = self.get_input(batch, target)
        x_src_1 = self.get_input(batch, source[0])
        x_src_2 = self.get_input(batch, source[1])
        x_src_3 = self.get_input(batch, source[2])
        input=torch.concat([x_src_1,x_src_2,x_src_3],dim=1)
        skip_pass = 1

        if self.stage == 1: 
            xrec, qloss = self(x_tar)
        else:
            h1=self.encode(input[:,0])
            h2=self.encode(input[:,1])
            h3=self.encode(input[:,2])
            h_comp=self.encode_comp(input)
            h_concat=torch.concat([h1,h2,h3,h_comp],dim=1)
            h=self.caff(h_concat,src_idx)
            z_src, qloss, _ = self.quantizer(h)
            z_tar_rec = self.spade(z_src, target)
            z_temp=self.encode(x_tar)
            z_tar,_,_=self.quantizer(z_temp)
            x_tar = z_tar
            xrec = z_tar_rec

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x_tar, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), skip_pass=skip_pass, split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x_tar, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), skip_pass=skip_pass, split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        target = random.choice(self.modalities)
        source = [m for m in self.modalities if m != target]
        src_idx=self.modalities_to_indices(source)
        x_tar = self.get_input(batch, target)
        x_src_1 = self.get_input(batch, source[0])
        x_src_2 = self.get_input(batch, source[1])
        x_src_3 = self.get_input(batch, source[2])
        input=torch.concat([x_src_1,x_src_2,x_src_3],dim=1)
       

        if self.stage == 1: 
            xrec, qloss = self(x_tar)
        else:
            h1=self.encode(input[:,0])
            h2=self.encode(input[:,1])
            h3=self.encode(input[:,2])
            h_comp=self.encode_comp(input)
            h_concat=torch.concat([h1,h2,h3,h_comp],dim=1)
            h=self.caff(h_concat,src_idx)
            z_src, qloss, _ = self.quantizer(h)
            z_tar_rec = self.spade(z_src, target)
            z_temp=self.encode(x_tar)
            z_tar,_,_=self.quantizer(z_temp)
            x_tar = z_tar
            xrec = z_tar_rec

        aeloss, log_dict_ae = self.loss(qloss, x_tar, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x_tar, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.stage == 1:
            for p in self.spade.parameters(): p.requires_grad = False
            for block in self.attn_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            for p in self.encoder_complementary.parameters(): p.requires_grad = False
            for p in self.conv1.parameters(): p.requires_grad = False
            for p in self.encoder.parameters(): p.requires_grad = True
            for p in self.decoder.parameters(): p.requires_grad = True
            opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()) +
                                  list(self.conv_out_enc.parameters()), lr=lr, betas=(0.5, 0.9))
        else:
            for p in self.spade.parameters(): p.requires_grad = True
            for p in self.encoder.parameters(): p.requires_grad = False
            for p in self.decoder.parameters(): p.requires_grad = False
            opt_ae = torch.optim.Adam(list(self.spade.parameters()),list(self.encoder_complementary.parameters()),list(self.conv1.parameters()),list(self.attn_blocks.parameters()), lr=lr, betas=(0.5, 0.9))

        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        target = random.choice(self.modalities)
        source = [m for m in self.modalities if m != target]
        src_idx=self.modalities_to_indices(source)
        x_tar = self.get_input(batch, target)
        x_src_1 = self.get_input(batch, source[0])
        x_src_2 = self.get_input(batch, source[1])
        x_src_3 = self.get_input(batch, source[2])
        input=torch.concat([x_src_1,x_src_2,x_src_3],dim=1)




        x_tar = self.get_input(batch, target)
        x_tar = x_tar.to(self.device)
        if self.stage == 1: 
            target = None
            src_idx=None
            input=x_tar
        xrec, _ = self(input, target,src_idx)

        log["source"] = input
        log["target"] = x_tar
        if self.stage == 1: 
            log["recon"] = xrec
        else:
            log[f"recon_{source}_to_{target}"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv3d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class EMAVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )
        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           remap=remap)
    def configure_optimizers(self):
        lr = self.learning_rate
        #Remove self.quantize from parameter list since it is updated via EMA
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []                                           