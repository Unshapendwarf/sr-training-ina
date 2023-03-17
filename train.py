import copy
import torch
import numpy as np
import pandas as pd
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
  
  psnr_log = []

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
      print("[Epoch {}] PSNR: {}".format(epoch, total_psnr))
      # print("[Epoch {}] PSNR: {} SSIM: {}".format(epoch, total_psnr, total_ssim))
      trainer.save_model(opt.model_name+"_epoch_"+str(epoch))
      psnr_log.append((epoch, total_psnr))
    
  trainer.save_model(opt.model_name+"_last")
  return psnr_log


if __name__ == '__main__':
    
  cluster_psnr_log = []
  
  # train clustered model
  if opt.cluster:
    for cluster_idx in range(opt.cluster_num):
      cluster_opt = copy.deepcopy(opt)
      cluster_opt.input_path = os.path.join(opt.data_root, f'clustered/{cluster_idx}/LR')
      cluster_opt.target_path = os.path.join(opt.data_root, f'clustered/{cluster_idx}/HR')
      cluster_opt.model_save_root = os.path.join(opt.log_dir, 'cluster')
      cluster_opt.model_name = opt.model_type + f'_cluster_{cluster_idx}'
      psnr_log = train_single(cluster_opt)
      cluster_psnr_log.append([x[1] for x in psnr_log])
  
  # train naive model
  opt.input_path = os.path.join(opt.data_root, 'LR')
  opt.target_path = os.path.join(opt.data_root, 'HR')
  opt.model_save_root = os.path.join(opt.log_dir, 'naive')
  opt.model_name = opt.model_type + f'_naive'
  opt.num_valid_image = opt.num_valid_image * opt.cluster_num
  naive_psnr_log = train_single(opt)
  naive_psnr_log = [x[1] for x in naive_psnr_log]
  
  # save psnr_log according to the epochs
  np_cluster_psnr = np.array(cluster_psnr_log)
  np_cluster_psnr = np.mean(np_cluster_psnr, axis=0)

  np_naive_psnr = np.array(naive_psnr_log)


  df_train_psnr = pd.DataFrame({'epoch': [x * opt.val_interval for x in range(len(np_cluster_psnr))] , 'clustered': np_cluster_psnr, 'naive': np_naive_psnr})
  if not os.path.exists(opt.result_root):
    os.makedirs(opt.result_root)

  df_train_psnr.to_excel(os.path.join(opt.result_root, 'training_psnr.xlsx'))

  print("\n\n" + "="*50)
  print("Training Finished")
  print("Naive psnr: {:.03f} \nClusterd psnr: {:.03f}".format(np_naive_psnr[-1], np_cluster_psnr[-1]))
  
  
  



