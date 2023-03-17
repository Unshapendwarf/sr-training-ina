import copy
import torch.nn as nn
import torch

from dataset import *
from option import opt
from trainer import Trainer


import model.ABPN as ABPN
from model.EDSR import SingleNetwork
import model.SAN.san as san
import model.RCAN.rcan as rcan

def test_single(opt):
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

  dataset = TestDataset(opt)
  trainer = Trainer(model, dataset, opt)
  psnr = trainer.test()
  avg_psnr = sum(psnr)/len(psnr)
  # avg_ssim = sum(ssim)/len(ssim)

  # print("[Result] PSNR: {} CNT {}".format(avg_psnr, len(psnr)))
  return avg_psnr, len(psnr)

if __name__ == '__main__':
  if opt.cluster:
    total_psnr = 0
    total_cnt = 0
    for cluster_idx in range(opt.cluster_num):
      cluster_opt = copy.deepcopy(opt)
      cluster_opt.input_path = os.path.join(opt.data_root, f'{cluster_idx}/LR')
      cluster_opt.target_path = os.path.join(opt.data_root, f'{cluster_idx}/HR')
      cluster_opt.pretrained_path = os.path.join(opt.pretrained_dir, f'{opt.model_type}_cluster_{cluster_idx}_last.pth')
      psnr, cnt = test_single(cluster_opt)
      total_psnr += psnr * cnt
      total_cnt += cnt
    print("[Result] PSNR: {}".format(total_psnr / total_cnt))
      
  else:
    opt.input_path = os.path.join(opt.data_root, 'LR')
    opt.target_path = os.path.join(opt.data_root, 'HR')
    opt.pretrained_path = os.path.join(opt.pretrained_dir, f'{opt.model_type}_last.pth')
    psnr, cnt = test_single(opt)
    print("[Result] PSNR: {}".format(psnr))