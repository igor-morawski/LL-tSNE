import os
import os.path as op
import glob
import numpy as np
import random
import torch

DEBUG = False
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp"]
CHECKPOINT_EXTENSIONS = ["pt", "pth"]
if DEBUG:
    RANDOM_SEED = 3
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def ensure_dir_exists(path):
    if not op.exists(path):
        os.mkdir(path)
    return True


def glob_extensions(directory, file_extensions, try_uppercase=True):
    result = []
    if type(file_extensions) == str:
        file_extensions = [file_extensions]
    for extension in file_extensions:
        result.extend(glob.glob(op.join(directory, "*.{}".format(extension))))
        if try_uppercase:
            result.extend(
                glob.glob(op.join(directory, "*.{}".format(extension.upper()))))
    return result

def get_state_dict(obj):
    x=obj
    if isinstance(x, str):
        x=torch.load(x)
    if 'state_dict' in x.keys():
        return x['state_dict']
    else:
        return x    


class ModuleWrapper(torch.nn.Module):
    def __init__(self, module, name):
        super(ModuleWrapper, self).__init__()
        self.name = name
        setattr(self, name, module)
    
    def forward(self, x):
        return getattr(self, self.name)(x)