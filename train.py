import torch
import torch.nn as nn


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

train_dataset = TrainDataset()
trainer = Trainer(model, train_dataset)


epoch = 0
while epoch < opt.num_epoch:
  trainer.train_one_epoch()
  total_sr_psnr = []
  total_sr_ssim = []
  total_latency = []
  for idx in range(opt.num_valid_image):
      if opt.is_split:
        sr_psnr, latency = trainer.validate_frame_split()
      else:
          sr_psnr, latency = trainer.validate_frame()         
      total_sr_psnr.append(sr_psnr)
      # total_sr_ssim.append(sr_ssim)
      total_latency.append(latency)
  total_psnr = np.mean(total_sr_psnr)
  # total_ssim = np.mean(total_sr_ssim)
  print("[Epoch {}] PSNR: {}".format(epoch, total_psnr))
  # print("[Epoch {}] PSNR: {} SSIM: {}".format(epoch, total_psnr, total_ssim))
  if epoch%5==0:
    trainer.save_model(opt.model_type+"_"+str(epoch))
  epoch +=1
trainer.save_model(opt.model_type+"_last")


