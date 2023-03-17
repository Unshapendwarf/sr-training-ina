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

def test_single(opt, cluster_idx=0):
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
  trainer = Trainer(model, dataset, opt, cluster_idx = cluster_idx)
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
      cluster_opt.input_path = os.path.join(opt.data_root, f'clustered/{cluster_idx}/LR')
      cluster_opt.target_path = os.path.join(opt.data_root, f'clustered/{cluster_idx}/HR')
      cluster_opt.pretrained_path = os.path.join(opt.pretrained_dir, f'cluster/{opt.model_type}_cluster_{cluster_idx}_last.pth')
      cluster_opt.img_save_dir = cluster_opt.img_save_dir + "/cluster"
      psnr, cnt = test_single(cluster_opt, cluster_idx)
      total_psnr += psnr * cnt
      total_cnt += cnt
    print("[Result] PSNR(Cluster):\t{}".format(total_psnr / total_cnt))
      
  # Test naive model
  opt.input_path = os.path.join(opt.data_root, 'LR')
  opt.target_path = os.path.join(opt.data_root, 'HR')
  opt.pretrained_path = os.path.join(opt.pretrained_dir, f'naive/{opt.model_type}_naive_last.pth')
  opt.img_save_dir = opt.img_save_dir + "/naive"
  psnr, cnt = test_single(opt)
  print("[Result] PSNR(Naive):\t{}".format(psnr))

