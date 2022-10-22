import os
import glob

root_dir = "/home/hong/1017_dir/sr-training"
target_dir = os.path.join(root_dir, "code/logs")
log_dir = os.path.join(target_dir, "*.log")


fname_logs = sorted(glob.glob(log_dir))
log_list = fname_logs

for filepath in log_list:
    pathname, ext = os.path.splitext(filepath)
    file_name = pathname.split("/")[-1]
    
    with open(filepath, "r") as fd_r:
        lines = fd_r.readlines()
        for line in lines:
            result = line.find("PSNR")
            if result != -1:
                print(f"{file_name} {line[14:-1]}")
    



# n_cls = 16
# for cls_idx in range(n_cls):
#     # Check whether the specified path exists or not
#     cls_dir = os.path.join(target_dir, f"{n_cls}cls/{cls_idx}")
#     isExist = os.path.exists(cls_dir)
#     if not isExist:
#     # Create a new directory because it does not exist
#         os.makedirs(cls_dir)
#         print("The new directory is created!")