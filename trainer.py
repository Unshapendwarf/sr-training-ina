import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import imageio
import time
import tqdm
from PIL import Image
# from sklearn.cluster import KMeans

import utility as util
from option import opt


img_splitter = util.ImageSplitter(opt.patch_size, opt.scale, opt.patch_size)

class Trainer():
    def __init__(self, model, dataset, opt, cluster_idx = 0, MLP=None):
        self.opt = opt
        self.model = model
        self.dataset = dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.timer = util.timer()
        self.epoch = 0
        self.model = self.model.to(self.device)
        self.MLP = MLP
        self.img_save = opt.img_save
        self.img_save_dir = self.opt.img_save_dir
        self.cluster_idx = cluster_idx

        if self.opt.pretrained:
          # print("Model loading...")
          self.model.load_state_dict(torch.load(self.opt.pretrained_path))

        if self.opt.img_save:
          if not os.path.isdir(self.opt.img_save_dir):
              os.makedirs(self.opt.img_save_dir)
        
        self.optimizer = optim.Adam(model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        self.loss_func = self._get_loss_func(self.opt.loss_type)
        
    def _get_loss_func(self, loss_type):
        if loss_type == 'l2':
            return nn.MSELoss()
        elif loss_type == 'l1':
            return nn.L1Loss()
        else:
            raise NotImplementedError

    def _adjust_learning_rate(self, epoch):
        if self.opt.lr_decay_epoch is not None:
            lr = self.opt.lr * (self.opt.lr_decay_rate ** (epoch // self.opt.lr_decay_epoch))
        else:
            lr = self.opt.lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_one_epoch(self):
        self.timer.tic()
        self.model.train()
        
        train_dataloader = DataLoader(dataset=self.dataset, num_workers=self.opt.num_thread, batch_size=self.opt.num_batch, pin_memory=True, shuffle=True)
        loss_total = []    
        for iteration, batch in enumerate(train_dataloader, 1):
            input, target = batch[0], batch[1]
            if self.opt.rgb_255:
              input = input*255.0
              target = target*255.0              
            input, target =  input.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_func(self.model(input), target)
            loss_total.append(loss.item())
            loss.backward()
            self.optimizer.step()           
            if iteration % 10 == 0:
                util.print_progress(iteration, len(self.dataset)/self.opt.num_batch, 'Train Progress (X{}):'.format(self.opt.scale), 'Complete', 1, 50)        
        print("[Epoch %d] loss %f" % (self.epoch, sum(loss_total)/len(loss_total)))
        
        self.epoch += 1
        self.epoch_elapsed = self.timer.toc_total_add()
        # print('Epoch[{}-train](complete): {}sec'.format(self.epoch, self.epoch_elapsed))
        

    def validate_frame_all(self):
        psnr_list = []
        with torch.no_grad():
          self.model.eval()
          for idx in range(len(self.dataset.lr_images)):
            input, target = self.dataset.getItemValidate_idx(idx) #returns PIL Image
            input_tensor = self.dataset.input_transform(input).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            target_tensor = self.dataset.input_transform(target).unsqueeze(0)
            target_tensor = torch.squeeze(torch.clamp(target_tensor, min=0, max=1.), 0).permute(1, 2, 0)
            target_tensor *= 255
            target = target_tensor.to(self.device)


            if self.opt.rgb_255:
              input_tensor = input_tensor*255.0

            t1 = time.time()
            output = self.model(input_tensor)
            torch.cuda.synchronize()
            t2 = time.time()

            if self.opt.rgb_255:
              output = torch.squeeze(torch.clamp(output, min=0, max=255.0), 0).permute(1, 2, 0)

            else:
              output = torch.squeeze(torch.clamp(output, min=0, max=1.), 0).permute(1, 2, 0)
              output *= 255
            # output_np = output.to('cpu').numpy()

            sr_psnr = util.gpu_psnr(output, target, max_value=255.0)
            psnr_list.append(sr_psnr)
          # sr_ssim = util.calculate_ssim(output_np, target_np)   


        return psnr_list, (t2-t1)*1000

    def validate_frame(self):

        with torch.no_grad():
            self.model.eval()
            input, target = self.dataset.getItemValidate() #returns PIL Image
            input_tensor = self.dataset.input_transform(input).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            target_tensor = self.dataset.input_transform(target).unsqueeze(0)
            target_tensor = torch.squeeze(torch.clamp(target_tensor, min=0, max=1.), 0).permute(1, 2, 0)
            target_tensor *= 255
            target = target_tensor.to(self.device)


            if self.opt.rgb_255:
              input_tensor = input_tensor*255.0

            t1 = time.time()
            output = self.model(input_tensor)
            torch.cuda.synchronize()
            t2 = time.time()

            if self.opt.rgb_255:
              output = torch.squeeze(torch.clamp(output, min=0, max=255.0), 0).permute(1, 2, 0)

            else:
              output = torch.squeeze(torch.clamp(output, min=0, max=1.), 0).permute(1, 2, 0)
              output *= 255
            # output_np = output.to('cpu').numpy()

            sr_psnr = util.gpu_psnr(output, target, max_value=255.0)
            # sr_ssim = util.calculate_ssim(output_np, target_np)   


        return sr_psnr, (t2-t1)*1000
        # return sr_psnr, sr_ssim, (t2-t1)*1000

    def validate_frame_split(self):

        with torch.no_grad():
            self.model.eval()
            input, target = self.dataset.getItemValidate() #returns PIL Image
            input_tensor = self.dataset.input_transform(input).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            target_tensor = self.dataset.input_transform(target).unsqueeze(0)
            target_tensor = torch.squeeze(torch.clamp(target_tensor, min=0, max=1.), 0).permute(1, 2, 0)
            target_tensor *= 255
            target = target_tensor.to(self.device)
            
            if self.opt.rgb_val == 255:
              input_tensor = input_tensor*255.0            
            t1 = time.time()
            tmp = img_splitter.split_img_tensor(input_tensor)
            res = []
            
            for i in range(len(tmp)):
              img_t = tmp[i]
              img_t = img_t.unsqueeze(0)
              img_t = img_t.to(device='cuda')
              a = self.model(img_t)
              a = a.squeeze()
              res.append(a)
            output = img_splitter.merge_img_tensor(res)
            t2 = time.time()

            if self.opt.rgb_255:
              output = torch.squeeze(torch.clamp(output, min=0, max=255.0), 0).permute(1, 2, 0)
            else:
              output = torch.squeeze(torch.clamp(output, min=0, max=1.), 0).permute(1, 2, 0)
              output *= 255

            # output_np = output.to('cpu').numpy()

            sr_psnr = util.get_psnr(output, target, max_value=255.0)
            # sr_ssim = util.calculate_ssim(output_np, target_np)   

        return sr_psnr, (t2-t1)*1000
        return sr_psnr, sr_ssim, (t2-t1)*1000
        
    def test(self):
        result_dir = self.opt.img_save_dir
        if not os.path.exists(result_dir):
          os.makedirs(result_dir)
        
        self.model.eval()
        data_loader = DataLoader(dataset=self.dataset, num_workers=self.opt.num_thread, batch_size=1, pin_memory=True, shuffle=False)
        total_psnr = []
        # total_ssim = []
        
        with torch.no_grad():
          for iteration, batch in enumerate(data_loader):
              input, target = batch[0], batch[1]
              input, target =  input.to(self.device), target.to(self.device)
              if self.opt.rgb_255:
                input = input*255.0
              output = self.model(input)
              if self.opt.rgb_255:
               output = torch.squeeze(torch.clamp(output, min=0, max=255.), 0).permute(1, 2, 0)               
              else:             
                output = torch.squeeze(torch.clamp(output, min=0, max=1.), 0).permute(1, 2, 0)
                output *= 255
              
              output_np = output.to('cpu').detach().numpy()
              
              # if self.img_save:
              #   output_np = output_np.astype(np.uint8)
              #   im = Image.fromarray(output_np)
              #   im.save(os.path.join(self.img_save_dir, '{:04d}.png'.format(self.cluster_idx*5 + iteration)))
              
              
              target = torch.squeeze(torch.clamp(target, min=0, max=1.), 0).permute(1, 2, 0)
              target *= 255
              # target_np = target.to('cpu').detach().numpy()  
              
              sr_psnr = util.gpu_psnr(output, target, max_value=255.0)
              # sr_psnr = util.get_psnr(output_np, target_np, max_value=255.0)
              # print(sr_psnr)
              # sr_ssim = util.calculate_ssim(output_np, target_np)   
              total_psnr.append(sr_psnr)
              # total_ssim.append(sr_ssim)
              if self.img_save:
                idx = str(self.cluster_idx * 5 + iteration).zfill(4)
                imageio.imwrite('{}/{}.png'.format(result_dir, idx), output_np.astype(np.uint8))

        return np.array(total_psnr)
        # return np.array(total_psnr), np.array(total_ssim)

    def test_analysis(self):
        result_dir = os.path.join('result', 'img')
        if not os.path.exists(result_dir):
          os.makedirs(result_dir)
        
        self.model.eval()
        data_loader = DataLoader(dataset=self.dataset, num_workers=self.opt.num_thread, batch_size=1, pin_memory=True, shuffle=False)
        total_psnr = []
        total_ssim = []
        
        with torch.no_grad():
          for iteration, batch in enumerate(data_loader):
              input, target = batch[0], batch[1]
              input, target =  input.to(self.device), target.to(self.device)
              if self.opt.rgb_255:
                input = input*255.0
              output = self.model(input)
              if self.opt.rgb_255:
               output = torch.squeeze(torch.clamp(output, min=0, max=255.), 0).permute(1, 2, 0)               
              else:             
                output = torch.squeeze(torch.clamp(output, min=0, max=1.), 0).permute(1, 2, 0)
                output *= 255
              output_np = output.to('cpu').detach().numpy()
              target = torch.squeeze(torch.clamp(target, min=0, max=1.), 0).permute(1, 2, 0)
              target *= 255
              target_np = target.to('cpu').detach().numpy()  
              sr_psnr = util.get_psnr(output_np, target_np, max_value=255.0)
              sr_ssim = util.calculate_ssim(output_np, target_np)   
              total_psnr.append(sr_psnr)
              total_ssim.append(sr_ssim)
              
        return np.array(total_psnr), np.array(total_ssim)

    def save_model(self, name):
        if not os.path.isdir(self.opt.model_save_root):
              os.makedirs(self.opt.model_save_root)
        save_path = os.path.join(self.opt.model_save_root, '{}.pth'.format(name))
        print(f"saving model to {save_path} ...")
        torch.save(self.model.state_dict(), save_path)