# SR-Traning
## Description
- Training and testing clustering for superesolution
- Upscaling Bicubic / Naive SR / Clustered SR

<!-- Model list: EDSR/[RCAN](https://github.com/yulunzhang/RCAN)/[SAN](https://github.com/daitao/SAN)/[ABPN](https://github.com/Holmes-Alan/ABPN) -->

## Installation

Installing environments...
```sh
pip3 install -r requirements.txt
```


## Executing
Train Cluster
```sh
./scripts/train_cluster.sh
```

Train Naive
```sh
./scripts/train_naive.sh
```

Test Cluster
```sh
./scripts/test_cluster.sh
```

Test Naive
```sh
./scripts/test_naive.sh
```


## Evaluation
Dataset: DIV2K x4 scale 
Epoch: 100
Number of iteration per epoch : 4000
|Method|PSNR(dB)|SSIM|
|------|---|---|
|Bicubic|35.02|0.9391|
|Naive SR|35.56|0.9406|
|Clustered SR|35.71|0.9428|



