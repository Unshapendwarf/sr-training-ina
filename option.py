import os, argparse

parser = argparse.ArgumentParser(description="Option for Super-resolution")

# Directory
parser.add_argument("--data_root", type=str, default="../data/frames")
parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument("--result_root", type=str, default="result")
parser.add_argument("--img_save_dir", type=str, default="result")

# Model Configuration
parser.add_argument("--model_type", type=str, required=True, choices=("EDSR", "RCAN", "ABPN", "SAN"))
parser.add_argument("--n_blocks", type=int, default=8)
parser.add_argument("--n_feats", type=int, default=64)
parser.add_argument("--n_groups", type=int, default=3)
parser.add_argument("--out_dim", type=int, default=32)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--pretrained_path", type=str)
parser.add_argument("--pretrained_dir", type=str)
parser.add_argument("--dev_path", type=str)
parser.add_argument("--img_save", action='store_true')

# Training
parser.add_argument("--num_batch", type=int, default=64)
parser.add_argument("--num_epoch", type=int, default=100)
parser.add_argument("--num_update_per_epoch", type=int, default=1000)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--loss_type", type=str, default="l1", choices=("l2", "l1"))
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--lr_decay_epoch", type=int, default=100)
parser.add_argument("--lr_decay_rate", type=float, default=1)
parser.add_argument("--num_valid_image", type=int, default=10, help="number of images used for validation")
parser.add_argument("--val_interval", type=int, default=10, help='validation interval')

# Dataset
parser.add_argument("--img_format", type=str, default="png", choices=("png", "AVIF"))
parser.add_argument("--rgb_255", action="store_true")


parser.add_argument("--patch_size", type=int, default=128)
parser.add_argument('--scale', type=int, required=True, help='target scales')

# Resource
parser.add_argument("--num_thread", type=int, default=0, help="number of threads used for loading data (used by DataLoader)")
parser.add_argument("--use_cuda", action="store_true", help="use GPU(s) for training")


# Clustering
parser.add_argument("--cluster", action='store_true')
parser.add_argument("--cluster_num", type=int, default=20, help='number of clusters')

opt = parser.parse_args()
opt.model_config = f"s{opt.scale}_{opt.n_blocks}b_{opt.n_feats}f"


if opt.model_type == "SAN" or opt.model_type == "ABPN":
  opt.is_split = True
else:
  opt.is_split = False
