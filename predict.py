import os

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#返回cuda表示成功
#或者
print(torch.cuda.is_available())