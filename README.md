# SR-Traning
## Description
- Training and testing clustering for superesolution
- Upscaling Bicubic / Naive SR / Clustered SR

<!-- Model list: EDSR/[RCAN](https://github.com/yulunzhang/RCAN)/[SAN](https://github.com/daitao/SAN)/[ABPN](https://github.com/Holmes-Alan/ABPN) -->

## Installation


Install anaconda
```sh
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
source ~/.bashrc
```

We recommend you to create conda env
```sh
conda create -n cluster_sr python=3.8 && conda activate cluster_sr
```

Clone the repository
```sh
git clone -b validation https://github.com/Unshapendwarf/sr-training-ina.git
cd sr-training-ina
```

Install required python dependences
```sh
pip3 install -r requirements.txt
```


## Dataset
You can download dataset [here](https://drive.google.com/file/d/1ussHhGVh0BEe_RjyGgD3lS3rJNwtOc4R/view?usp=sharing) or follow below scripts
```sh
python3 scripts/download_data.py
```

## Execution
Train SR
```sh
bash scripts/train.sh
```

Test SR
```sh
bash scripts/test.sh
```



## Evaluation
- Dataset: Custom youtube frames x4 scale 
- Epoch: 100
- Number of iteration per epoch : 300

|Method|PSNR(dB)|
|------|---|
|Bicubic|29.059|
|Naive SR|31.176|
|Clustered SR|32.459|



