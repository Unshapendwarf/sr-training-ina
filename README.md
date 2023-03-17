# SR-Traning
## Description
- Training and testing clustering for superesolution
- Upscaling Bicubic / Naive SR / Clustered SR

<!-- Model list: EDSR/[RCAN](https://github.com/yulunzhang/RCAN)/[SAN](https://github.com/daitao/SAN)/[ABPN](https://github.com/Holmes-Alan/ABPN) -->

## Installation

We recommend you to create conda env
```sh
conda create -n cluster_sr python=3.8
conda activate cluster_sr
```

Install required python dependences
```sh
pip3 install -r requirements.txt
```

Clone the repository
```sh
git clone https://github.com/Unshapendwarf/sr-training-ina.git
cd sr-training-ina
```

## Dataset
You can download dataset [here](https://drive.google.com/file/d/1ussHhGVh0BEe_RjyGgD3lS3rJNwtOc4R/view?usp=sharing) or follow below scripts
```sh
python3 scripts/download_data.py
```

## Execution
Train Cluster
```sh
bash scripts/train_cluster.sh
```

Train Naive
```sh
bash scripts/train_naive.sh
```

Test Cluster
```sh
bash scripts/test_cluster.sh
```

Test Naive
```sh
bash scripts/test_naive.sh
```


## Evaluation
Dataset: Custom youtube frames x4 scale 
Epoch: 100
Number of iteration per epoch : 300
|Method|PSNR(dB)|
|------|---|
|Bicubic|29.059|
|Naive SR|31.176|
|Clustered SR|32.459|



