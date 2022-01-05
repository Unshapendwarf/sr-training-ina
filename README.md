# SR-Traning
## Description
- Training and testing SR Model
- Model list: EDSR/[RCAN](https://github.com/yulunzhang/RCAN)/[SAN](https://github.com/daitao/SAN)/[ABPN](https://github.com/Holmes-Alan/ABPN)

## Installation

Installing environments...
```sh
conda install --yes --file requirements.txt
```


## Executing
Train
```sh
./train.sh
```
Test
```sh
./test.sh
```
## Evaluation
Dataset: DIV2K x4 scale 
Epoch: 100
Number of iteration per epoch : 4000
|Model|PSNR(dB)|SSIM|Model size(MB)|
|------|---|---|---|
|EDSR|35.02|0.9391|2.93|
|RCAN|35.56|0.9406|7.98|
|SAN|35.71|0.9428|7.96|
|ABPN|34.25|0.9351|5.05|



