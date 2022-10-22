import os

root_dir = "/home/hong/1017_dir/sr-training"
target_dir = os.path.join(root_dir, "cls_models")

n_cls = 20
for cls_idx in range(n_cls):
    # Check whether the specified path exists or not
    cls_dir = os.path.join(target_dir, f"oneh_{n_cls}cls300/{cls_idx}")
    isExist = os.path.exists(cls_dir)
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(cls_dir)
        print("The new directory is created!")