import copy
import torch
import torch.nn as nn


from dataset import *
from option import opt
from trainer import Trainer


import model.ABPN as ABPN
from model.EDSR import SingleNetwork
import model.SAN.san as san
import model.RCAN.rcan as rcan

def train_single(opt):
  if opt.model_type == "EDSR":
    model = SingleNetwork(num_block=opt.n_blocks, num_feature=opt.n_feats, num_channel=3, scale=opt.scale, bias=True)
  elif opt.model_type == "RCAN":
    model = rcan.RCAN(opt.n_groups, opt.n_blocks, opt.n_feats, opt.n_feats//2, opt.scale)
  elif opt.model_type == "SAN":
    model = san.SAN(opt.n_groups, opt.n_blocks, opt.n_feats, opt.n_feats//2, opt.scale)
  elif opt.model_type == "ABPN":  
    model = ABPN.ABPN_v7(input_dim=3, dim=opt.out_dim)
  
  train_dataset = TrainDataset(opt)
  trainer = Trainer(model, train_dataset, opt)
      
  for epoch in range(opt.num_epoch):
    trainer.train_one_epoch()
    total_sr_psnr = []
    total_sr_ssim = []
    total_latency = []
    
    if epoch % opt.val_interval == (opt.val_interval - 1):
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
      print("[Epoch {}] PSNR: {}".format(epoch+1, total_psnr))
      # print("[Epoch {}] PSNR: {} SSIM: {}".format(epoch, total_psnr, total_ssim))
      trainer.save_model(opt.model_name+"_epoch_"+str(epoch+1))
    
  trainer.save_model(opt.model_name+"_last")


if __name__ == '__main__':
      
  if opt.cluster:
    for cluster_idx in range(opt.cluster_num):
      cluster_opt = copy.deepcopy(opt)
      cluster_opt.input_path = os.path.join(opt.data_root, f'{cluster_idx}/LR')
      cluster_opt.target_path = os.path.join(opt.data_root, f'{cluster_idx}/HR')
      cluster_opt.model_name = opt.model_type + f'_cluster_{cluster_idx}'
      train_single(cluster_opt)
  else:
    opt.input_path = os.path.join(opt.data_root, 'LR')
    opt.target_path = os.path.join(opt.data_root, 'HR')
    opt.model_name = opt.model_type
    train_single(opt)



