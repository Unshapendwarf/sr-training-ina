from logging import root
import os
import glob

root_dir = "/home/hong/dataset/oneh_data"
target_dir = os.path.join(root_dir, "20cls")

hr_list = glob.glob(root_dir+ '/HR/*png')
lr_list = glob.glob(root_dir+ '/LR/*png')
hr_list.sort()
lr_list.sort()


n_cls = 20
for cls_idx in range(n_cls):
    # Check whether the specified path exists or not
    cls_dir_hr = os.path.join(target_dir, f"{cls_idx}/HR")
    cls_dir_lr = os.path.join(target_dir, f"{cls_idx}/LR")
    isExist = os.path.exists(cls_dir_hr)
    isExist = os.path.exists(cls_dir_hr)
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(cls_dir_hr)
        os.makedirs(cls_dir_lr)
        print("The new directory is created!")

    for img in hr_list[cls_idx*5:cls_idx*5+5]:
        cmd = f'cp {img} {cls_dir_hr}'
        os.system(cmd)

    for img in lr_list[cls_idx*5:cls_idx*5+5]:
        cmd = f'cp {img} {cls_dir_lr}'
        os.system(cmd)



# lis = os.path.join(root_dir, "HR", "*.png")
