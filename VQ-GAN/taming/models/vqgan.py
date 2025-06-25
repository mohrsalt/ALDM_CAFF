import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import copy
from main import instantiate_from_config

from taming.models.normalization import SPADEGenerator
from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer


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
        
        return w
    
class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 num_classes=None, # list of modalities to use
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 stage=1,
                 ):
        super().__init__()
        self.automatic_optimization = False

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
        self.attn_blocks = nn.ModuleList([CHattnblock(64*2) for i in range(5)]) #new
        self.conv1 = nn.Conv3d(256, 128, 1) #new
        
        
        self.spade = SPADEGenerator(num_classes, ddconfig["z_channels"])
        self.conv_out_enc = torch.nn.Conv3d(128,
                                        4,
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
            h1=self.encode(input[:,0].unsqueeze(1))
            h2=self.encode(input[:,1].unsqueeze(1))
            h3=self.encode(input[:,2].unsqueeze(1))
            h_comp=self.encoder_complementary(input)
            h1=h1.unsqueeze(1)
            h2=h2.unsqueeze(1)
            h3=h3.unsqueeze(1)
            h_comp=h_comp.unsqueeze(1)
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
        B=net_z.shape[0]  
   
        rawraw_attns=[]
        for b in range(B):
            sample_attns=[]
            for i, src in enumerate(chosen_sources[b,:]):
                attn_map = self.attn_blocks[src](net_z[b, i].unsqueeze(0)) 
                sample_attns.append(attn_map.unsqueeze(1)) 
            sample_attns = torch.cat(sample_attns, dim=1) 
            rawraw_attns.append(sample_attns)
        raw_attns.append(torch.cat(rawraw_attns,dim=0))
    
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
    
    def modalities_to_indices(self, source):
        m = ["t1n", "t1c", "t2w", "t2f"]
     
        if isinstance(source[0], str):  # handle batch size 1
            source = [source]

        choices = []
        for s in source:
            # flatten if s contains tuples
            if isinstance(s[0], tuple):
                s = [mod[0] for mod in s]
            choices.append([m.index(mod) for mod in s])

        batched_choices = torch.tensor(choices)

        return batched_choices



    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def training_step(self, batch, batch_idx):
        # MANUAL optimization mode
        opt_ae, opt_disc = self.optimizers()
        flat_sources = [s[0] if isinstance(s, tuple) else s for s in batch["sources_list"]]
        src_idx=self.modalities_to_indices(flat_sources)
        x_tar = self.get_input(batch, "target")

        input=self.get_input(batch, "source")
        y = batch["target_class"].long()
        
        
        skip_pass = 1
      
        if self.stage == 1:
            xrec, qloss = self(x_tar)
            y=None
        else:
            h1 = self.encode(input[:, 0].unsqueeze(1))
            
            h2 = self.encode(input[:, 1].unsqueeze(1))
            h3 = self.encode(input[:, 2].unsqueeze(1))
            h_comp = self.encode_comp(input)
            h1=h1.unsqueeze(1)
            h2=h2.unsqueeze(1)
            h3=h3.unsqueeze(1)
            h_comp=h_comp.unsqueeze(1)
            h_concat = torch.concat([h1, h2, h3, h_comp], dim=1)
            h = self.caff(h_concat, src_idx)
            z_src, qloss, _ = self.quantizer(h)
            z_tar_rec = self.spade(z_src, y)
            z_temp = self.encode(x_tar)
            z_tar, _, _ = self.quantizer(z_temp)
            x_tar = z_tar
            xrec = z_tar_rec

        # ---- Autoencoder Update ----
     
        opt_ae.zero_grad()
      
        
        aeloss, log_dict_ae = self.loss(
            qloss, x_tar, xrec, 0, self.global_step,
            last_layer=self.get_last_layer(), label=y, split="train"
        )
  
        self.manual_backward(aeloss)

        opt_ae.step()
 
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # ---- Discriminator Update ----
        opt_disc.zero_grad()
        discloss, log_dict_disc = self.loss(
            qloss, x_tar, xrec, 1, self.global_step,
            last_layer=self.get_last_layer(), label=y, split="train"
        )
        self.manual_backward(discloss)
        opt_disc.step()

        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return aeloss + discloss

    def validation_step(self, batch, batch_idx):
        flat_sources = [s[0] if isinstance(s, tuple) else s for s in batch["sources_list"]]

        src_idx=self.modalities_to_indices(flat_sources)
        x_tar = self.get_input(batch, "target")

        input=self.get_input(batch, "source")
        y = batch["target_class"].long()
       

        if self.stage == 1: 
            xrec, qloss = self(x_tar)
            
        else:
            h1=self.encode(input[:,0].unsqueeze(1))
            h2=self.encode(input[:,1].unsqueeze(1))
            h3=self.encode(input[:,2].unsqueeze(1))
            h_comp=self.encode_comp(input)
            h1=h1.unsqueeze(1)
            h2=h2.unsqueeze(1)
            h3=h3.unsqueeze(1)
            h_comp=h_comp.unsqueeze(1)
            
            h_concat=torch.concat([h1,h2,h3,h_comp],dim=1)
            h=self.caff(h_concat,src_idx)
            z_src, qloss, _ = self.quantizer(h)
            z_tar_rec = self.spade(z_src, y)
            z_temp=self.encode(x_tar)
            z_tar,_,_=self.quantizer(z_temp)
            x_tar = z_tar
            xrec = z_tar_rec
        print("xrec shape: ",xrec.shape)
        aeloss, log_dict_ae = self.loss(qloss, x_tar, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x_tar, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae.pop("val/rec_loss", None)
        
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
            params = (
    list(self.spade.parameters()) +
    list(self.encoder_complementary.parameters()) +
    list(self.conv1.parameters()) +
    list(self.attn_blocks.parameters())
)

            opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))

        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        flat_sources = [s[0] if isinstance(s, tuple) else s for s in batch["sources_list"]]
        src_idx=self.modalities_to_indices(flat_sources)
        x_tar = self.get_input(batch, "target")

        input=self.get_input(batch, "source").to(self.device)
        y = batch["target_class"].long()
        target=batch["t_list"]




       
        x_tar = x_tar.to(self.device)
        if self.stage == 1: 
            y = None
            src_idx=None
            input=x_tar
        xrec, _ = self(input, y,src_idx)

        log["source"] = input
        log["target"] = x_tar
        if self.stage == 1: 
            log["recon"] = xrec
        else:
            log[f"recon_{batch['sources_list']}_to_{target}"] = xrec
        return log                                         