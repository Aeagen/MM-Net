import os
from pathlib import Path

# ----------------------------2020----------------------------
# user = "YOU"
# BRATS_TRAIN_FOLDERS = f"/home/wpx/wpx/2020brain/train"
# BRATS_TRAIN_valid_FOLDERS = f"/home/laisong/awpx/2020brain/train"
# # BRATS_VAL_FOLDER = f"/home/laisong/awpx/2021brain/2021_train"
# BRATS_VAL_FOLDER = f"/home/wpx/wpx/2020brain/valid"
# BRATS_TEST_FOLDER = f"/home/{user}/Datasets/brats2020/MICCAI_BraTS2020_TestingData"


# --------------------------2018brain-----------------
user = "YOU"
BRATS_TRAIN_FOLDERS = f"/home/wpx/wpx/2018brain/train"
BRATS_TRAIN_valid_FOLDERS = f"/home/wpx/wpx/2020brain/train"
BRATS_VAL_FOLDER = f"/home/wpx/wpx/2018brain/valid"
BRATS_TEST_FOLDER = f"/home/{user}/Datasets/brats2020/MICCAI_BraTS2020_TestingData"
#-------------------------------------------------------------

# --------------------------2019brain-----------------
# user = "YOU"
# BRATS_TRAIN_FOLDERS = f"/home/wpx/wpx/2019brain/train"
# BRATS_TRAIN_valid_FOLDERS = f"/home/wpx/wpx/2020brain/train"
# BRATS_VAL_FOLDER = f"/home/wpx/wpx/2019brain/valid"
# BRATS_TEST_FOLDER = f"/home/{user}/Datasets/brats2020/MICCAI_BraTS2020_TestingData"
#-------------------------------------------------------------

# --------------------------2020brain-----------------
# user = "YOU"
# BRATS_TRAIN_FOLDERS = f"/home/wpx/wpx/2020brain/train"
# BRATS_TRAIN_valid_FOLDERS = f"/home/wpx/wpx/2020brain/train"
# BRATS_VAL_FOLDER = f"/home/wpx/wpx/2020brain/valid"
# BRATS_TEST_FOLDER = f"/home/{user}/Datasets/brats2020/MICCAI_BraTS2020_TestingData"
#-------------------------------------------------------------

def get_brats_folder(on="val"):
    if on == "train":
        return os.environ['BRATS_FOLDERS'] if 'BRATS_FOLDERS' in os.environ else BRATS_TRAIN_FOLDERS
    elif on == "train_val":
        return os.environ['BRATS_FOLDERS'] if 'BRATS_FOLDERS' in os.environ else BRATS_TRAIN_valid_FOLDERS
    elif on == "val":
        return os.environ['BRATS_VAL_FOLDER'] if 'BRATS_VAL_FOLDER' in os.environ else BRATS_VAL_FOLDER
    elif on == "test":
        return os.environ['BRATS_TEST_FOLDER'] if 'BRATS_TEST_FOLDER' in os.environ else BRATS_TEST_FOLDER
