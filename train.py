import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        model = rcan.RCAN(opt.n_groups, opt.n_blocks, opt.n_feats, opt.n_feats // 2, opt.scale)
    elif opt.model_type == "SAN":
        model = san.SAN(opt.n_groups, opt.n_blocks, opt.n_feats, opt.n_feats // 2, opt.scale)
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

        if epoch % opt.val_interval == (opt.val_interval - 1) or epoch == 0:
            # for idx in range(opt.num_valid_image):
            #     if opt.is_split:
            #       sr_psnr, latency = trainer.validate_frame_split()
            #     else:
            #       sr_psnr, latency = trainer.validate_frame_all()
            #     total_sr_psnr.append(sr_psnr)
            #     # total_sr_ssim.append(sr_ssim)
            #     total_latency.append(latency)
            total_sr_psnr, latency = trainer.validate_frame_all()
            total_psnr = np.mean(total_sr_psnr)
            # total_ssim = np.mean(total_sr_ssim)
            print("[Epoch {}] PSNR: {}".format(epoch, total_psnr))
            # print("[Epoch {}] PSNR: {} SSIM: {}".format(epoch, total_psnr, total_ssim))
            trainer.save_model(opt.model_name + "_epoch_" + str(epoch))
            psnr_log.append((epoch, total_psnr))

    trainer.save_model(opt.model_name + "_last")
    return psnr_log


def train_all(opt):
    # check naive method
    bicubic_opt = copy.deepcopy(opt)
    bicubic_opt.input_path = os.path.join(opt.data_root, "bicubic")
    bicubic_opt.target_path = os.path.join(opt.data_root, "HR")
    bicubic_opt.scale = 1
    
    print("Preparing HR/LR dataset for training ... ")
    
    bicubic_dataset = TestDataset(bicubic_opt)
    bicubic_psnr = validate_raw(bicubic_dataset)

    # model setup
    if opt.model_type == "EDSR":
        model = SingleNetwork(num_block=opt.n_blocks, num_feature=opt.n_feats, num_channel=3, scale=opt.scale, bias=True)
    elif opt.model_type == "RCAN":
        model = rcan.RCAN(opt.n_groups, opt.n_blocks, opt.n_feats, opt.n_feats // 2, opt.scale)
    elif opt.model_type == "SAN":
        model = san.SAN(opt.n_groups, opt.n_blocks, opt.n_feats, opt.n_feats // 2, opt.scale)
    elif opt.model_type == "ABPN":
        model = ABPN.ABPN_v7(input_dim=3, dim=opt.out_dim)

    # opt setup
    opt_list = []
    if opt.cluster:
        for cluster_idx in range(opt.cluster_num):
            cluster_opt = copy.deepcopy(opt)
            cluster_opt.input_path = os.path.join(opt.data_root, f"clustered/{cluster_idx}/LR")
            cluster_opt.target_path = os.path.join(opt.data_root, f"clustered/{cluster_idx}/HR")
            cluster_opt.model_save_root = os.path.join(opt.log_dir, "cluster")
            cluster_opt.model_name = opt.model_type + f"_cluster_{cluster_idx}"
            opt_list.append(cluster_opt)
    naive_opt = copy.deepcopy(opt)
    naive_opt.input_path = os.path.join(opt.data_root, "LR")
    naive_opt.target_path = os.path.join(opt.data_root, "HR")
    naive_opt.model_save_root = os.path.join(opt.log_dir, "naive")
    naive_opt.model_name = opt.model_type + f"_naive"
    naive_opt.num_valid_image = opt.num_valid_image * opt.cluster_num
    opt_list.append(naive_opt)

    # trainer setup
    cluster_psnr_log = []
    trainer_list = []
    for cur_opt in opt_list:
        train_dataset = TrainDataset(cur_opt)
        trainer = Trainer(model, train_dataset, cur_opt)
        trainer_list.append(trainer)
        cluster_psnr_log.append([])

    # train
    with tqdm(range(opt.num_epoch), unit="epoch") as tepoch:
        for epoch in tepoch:
            for i, cur_data in enumerate(zip(trainer_list, opt_list)):
                cur_trainer, cur_opt = cur_data
                cur_trainer.train_one_epoch()
                total_sr_psnr = []
                total_sr_ssim = []
                total_latency = []

                if epoch % cur_opt.val_interval == (cur_opt.val_interval - 1) or epoch == 0:
                    # for idx in range(cur_opt.num_valid_image):
                    #     if cur_opt.is_split:
                    #       sr_psnr, latency = cur_trainer.validate_frame_split()
                    #     else:
                    #       sr_psnr, latency = cur_trainer.validate_frame_all()
                    #     total_sr_psnr.append(sr_psnr)
                    #     # total_sr_ssim.append(sr_ssim)
                    #     total_latency.append(latency)
                    total_sr_psnr, latency = cur_trainer.validate_frame_all()
                    total_psnr = np.mean(total_sr_psnr)
                    # total_ssim = np.mean(total_sr_ssim)
                    # print("[Epoch {}] PSNR: {}".format(epoch, total_psnr))
                    # print("[Epoch {}] PSNR: {} SSIM: {}".format(epoch, total_psnr, total_ssim))
                    cur_trainer.save_model(cur_opt.model_name + "_epoch_" + str(epoch))
                    if epoch == (cur_opt.num_epoch - 1):
                        cur_trainer.save_model(cur_opt.model_name + "_last")
                    cluster_psnr_log[i].append(total_psnr)

            # plot figure 
            if epoch % cur_opt.val_interval == (cur_opt.val_interval -1 ) or epoch == 0:    
                np_cluster_psnr = np.array(cluster_psnr_log[:-1])
                np_cluster_psnr_mean = np.mean(np_cluster_psnr, axis=0)
                # print(np_cluster_psnr_mean)
                
                np_naive_psnr = np.array(cluster_psnr_log[-1])
                np_bicubic_psnr = np.array([bicubic_psnr] * len(np_cluster_psnr_mean))
                # print(np_naive_psnr)

                df_train_psnr = pd.DataFrame(
                    {"clustered": np_cluster_psnr_mean, "naive": np_naive_psnr, "bicubic": np_bicubic_psnr},
                    index=[x * cur_opt.val_interval for x in range(len(np_cluster_psnr_mean))],
                )
                if not os.path.exists(cur_opt.result_root):
                    os.makedirs(cur_opt.result_root)

                df_train_psnr.to_excel(os.path.join(cur_opt.result_root, f"training_psnr_{epoch}.xlsx"))

                plt = df_train_psnr.plot(title="Training PSNR", lw=2, marker=".")
                plt.set_ylabel("PSNR")
                plt.set_xlabel("Epoch")
                fig = plt.get_figure()
                fig.savefig(os.path.join(cur_opt.result_root, f"training_psnr_{epoch}.png"))

                tepoch.set_postfix(naiv_psnr=np_naive_psnr[-1], cls_pnsr=np_cluster_psnr_mean[-1])

    print("\n\n" + "=" * 50)
    print("Training Finished")
    print("Bicubic psnr:{:.03f} \nNaive psnr: {:.03f} \nClusterd psnr: {:.03f}".format(np_bicubic_psnr[-1], np_naive_psnr[-1], np_cluster_psnr_mean[-1]))


if __name__ == "__main__":
    train_all(opt)
