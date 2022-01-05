import torch.nn as nn
import torch

from dataset import *
from option import opt
from trainer import Trainer


import model.ABPN as ABPN
from model.EDSR import SingleNetwork
import model.SAN.san as san
import model.RCAN.rcan as rcan


if opt.model_type == "EDSR":
  model = SingleNetwork(num_block=opt.n_blocks, num_feature=opt.n_feats, num_channel=3, scale=opt.scale, bias=True)
elif opt.model_type == "RCAN":
  model = rcan.RCAN(opt.n_groups, opt.n_blocks, opt.n_feats, opt.n_feats//2, opt.scale)
elif opt.model_type == "SAN":
  model = san.SAN(opt.n_groups, opt.n_blocks, opt.n_feats, opt.n_feats//2, opt.scale)
elif opt.model_type == "ABPN":  
  model = ABPN.ABPN_v7(input_dim=3, dim=opt.out_dim)


if not opt.pretrained:
  print("please load the model file")
  exit(1)

dataset = TestDataset()
trainer = Trainer(model, dataset)
psnr, ssim = trainer.test()
avg_psnr = sum(psnr)/len(psnr)
avg_ssim = sum(ssim)/len(ssim)

print("[Result] PSNR: {} SSIM {}".format(avg_psnr, avg_ssim))