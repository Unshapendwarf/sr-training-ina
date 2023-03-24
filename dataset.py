import glob, random, os, random, sys
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor
import numpy as np
# import pillow_avif
from tqdm import tqdm
import pandas as pd
import random
import csv

import utility as util
# from option import opt

  

class TrainDataset(data.Dataset):
    def __init__(self, opt):
        super(TrainDataset, self).__init__()
        self.opt = opt
        assert os.path.exists(self.opt.input_path)
        assert os.path.exists(self.opt.target_path)
        self.img_format = self.opt.img_format
        self.input_transform = Compose([ToTensor(),])
        self._setup()


    def _setup(self): 

        self.lr_filenames = glob.glob(f'{self.opt.input_path}/*.' + self.opt.img_format)
        self.hr_filenames = glob.glob(f'{self.opt.target_path}/*.' + self.opt.img_format)
   
        assert(len(self.lr_filenames) == len(self.hr_filenames))  
        
        self.lr_filenames.sort(key=util.natural_keys)
        self.hr_filenames.sort(key=util.natural_keys)



        self.f_len = len(self.lr_filenames)

        self.lr_images = []
        self.hr_images = []
        
        print("Preparing HR/LR dataset for training ... ")
        # for lr, hr in tqdm(zip(self.lr_filenames, self.hr_filenames), total=len(self.lr_filenames)):
        for lr, hr in zip(self.lr_filenames, self.hr_filenames):
            lr_img = Image.open(lr)
            hr_img = Image.open(hr)
            lr_img.load()
            hr_img.load()
            lr_w, lr_h = lr_img.size
            hr_w, hr_h = hr_img.size

            assert(lr_w * self.opt.scale == hr_w and lr_h * self.opt.scale == hr_h)
            self.lr_images.append(lr_img)
            self.hr_images.append(hr_img)
            
    
            
    def getItemTrain(self, idx):
        input = self.lr_images[idx]
        target = self.hr_images[idx]
        
        #Randomly select crop location
       
        width, height = input.size
        height_ = random.randrange(0, height - self.opt.patch_size + 1)
        width_ = random.randrange(0, width - self.opt.patch_size + 1)
        input = input.crop((width_ , height_, width_ + self.opt.patch_size, height_ + self.opt.patch_size))
        target = target.crop((width_ * self.opt.scale, height_ * self.opt.scale, (width_ + self.opt.patch_size) * self.opt.scale, (height_ + self.opt.patch_size) * self.opt.scale))
      
        input = self.input_transform(input)
        target = self.input_transform(target)
        return input, target

    def getItemValidate(self):

        idx = random.randint(0, self.f_len-1)

        input = self.lr_images[idx]
        target = self.hr_images[idx]

        return input, target

    def getItemValidate_idx(self, idx):
        input = self.lr_images[idx]
        target = self.hr_images[idx]

        return input, target

    def getDatasetLen(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.getItemTrain(idx % self.f_len)                

    def __len__(self):
          return self.opt.num_batch * self.opt.num_update_per_epoch

class TestDataset(data.Dataset):
    def __init__(self, opt):
        super(TestDataset, self).__init__()
        self.opt = opt
        assert os.path.exists(self.opt.input_path)
        assert os.path.exists(self.opt.target_path)
        self.input_transform = Compose([ToTensor(),])
        self._setup()
        
    def _setup(self):
    
        self.lr_filenames = glob.glob(f'{self.opt.input_path}/*.' + self.opt.img_format)
        self.hr_filenames = glob.glob(f'{self.opt.target_path}/*.' + self.opt.img_format)
   
        assert(len(self.lr_filenames) == len(self.hr_filenames))

        self.lr_filenames.sort(key=util.natural_keys)
        self.hr_filenames.sort(key=util.natural_keys)

        self.lr_images = []
        self.hr_images = []
               
        # print("Preparing HR/LR dataset for training ... ")
        for lr, hr in zip(self.lr_filenames, self.hr_filenames):
            lr_img = Image.open(lr)
            hr_img = Image.open(hr)
            lr_img.load()
            hr_img.load()
            lr_w, lr_h = lr_img.size
            hr_w, hr_h = hr_img.size

            assert(lr_w * self.opt.scale == hr_w and lr_h * self.opt.scale == hr_h)
            self.lr_images.append(lr_img)
            self.hr_images.append(hr_img)

    def __len__(self):
          return len(self.hr_filenames)

    def __getitem__(self, idx):
        input = self.lr_images[idx]
        target = self.hr_images[idx]

        input = self.input_transform(input)
        target = self.input_transform(target)
        return input, target
