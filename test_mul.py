import torch.nn as nn
import torch

from dataset import *
from option import opt
from trainer_mul import Trainer


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

pretrained_paths = [ os.path.join(opt.dev_path, f"EDSR_{u_epoch}.pth") for u_epoch in [0, 10, 20, 50, 100]]

# pretrained_paths = [ os.path.join(opt.dev_path, f"EDSR_{u_epoch}.pth") for u_epoch in [0, 10, 20, 50, 100, 150, 195, 295]]
dataset = TestDataset()
for k, pr_path in enumerate(pretrained_paths):
  trainer = Trainer(model, dataset, pr_path)
  psnr = trainer.test()
  avg_psnr = sum(psnr)/len(psnr)
  # avg_ssim = sum(ssim)/len(ssim)
  if k==len(pretrained_paths)-1:
    print("[RESULT] PSNR: {} CNT {}".format(avg_psnr, len(psnr)))
    break
  print("[RESULT] PSNR: {} CNT {}".format(avg_psnr, len(psnr)), end=" / ")