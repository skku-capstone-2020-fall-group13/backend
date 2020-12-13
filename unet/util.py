import os
import random
import torch
import logging
import numpy as np

color2idx = {
    (0, 0, 0): 0,
    (255, 255, 0): 1,
    (150, 80, 0): 2,
    (100, 100, 100): 3,
    (0, 0, 150): 4,
    (0, 255, 0): 5,
    (0, 125, 0): 6,
    (150, 150, 250): 7,
    (255, 255, 255): 8
}

idx2color = { v: k for k, v in color2idx.items() }

def init_logger():
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cat_to_rgb(cat_arr):
    arr = np.zeros((*cat_arr.shape[:2], 3), dtype=np.int)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if cat_arr[i,j] in idx2color:
                arr[i,j] = idx2color[cat_arr[i,j]]
            else:
                arr[i,j] = (255, 255, 255)
    
    return arr

def rgb_to_cat(rgb_arr):
    arr = np.zeros(rgb_arr.shape[:2], dtype=np.int64)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if tuple(rgb_arr[i,j]) in color2idx:
                arr[i,j] = color2idx[tuple(rgb_arr[i,j])]
            else:
                arr[i,j] = 8
        
    return arr